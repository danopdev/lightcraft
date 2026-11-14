//! Instance segmentation for user-converted YOLO ONNX models.

use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;
use std::sync::Arc;
use tract_onnx::prelude::{Framework, InferenceModelExt, IntoRunnable, IntoTValue, Tensor, TypedRunnableModel, onnx};
use tract_onnx::tract_core::internal::tvec;

use crate::{Click, Error, MASK_SIDE, Prediction, Result};

const MODEL_FILE: &str = "model.onnx";
const MANIFEST_FILE: &str = "lightcraft-yolo.json";
const SCHEMA: &str = "lightcraft-yolo-seg";
const MAX_MANIFEST_BYTES: u64 = 1 << 20;
const MAX_MODEL_BYTES: u64 = 1 << 30;
const INPUT_EDGE: usize = 640;
const MAX_INPUT_BYTES: usize = 256 << 20;
const MAX_OUTPUT_VALUES: usize = 20_000_000;
const MAX_LABELS: usize = 1024;
const MAX_DETECTIONS: usize = 300;
const CONFIDENCE: f32 = 0.25;
const NMS_IOU: f32 = 0.5;
const OUTSIDE_LOGIT: f32 = -16.0;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: String,
    version: u32,
    image_size: usize,
    labels: Vec<String>,
}

struct Output {
    shape: Vec<usize>,
    values: Vec<f32>,
}

struct Prototypes {
    values: Vec<f32>,
    channels: usize,
    height: usize,
    width: usize,
}

#[derive(Clone, Copy)]
enum RawLayout {
    ChannelsFirst,
    ChannelsLast,
}

#[derive(Clone)]
struct Detection {
    label: usize,
    score: f32,
    bounds: [f32; 4],
    coefficients: Vec<f32>,
}

struct Transform {
    source_width: usize,
    source_height: usize,
    scale_x: f32,
    scale_y: f32,
    pad_x: usize,
    pad_y: usize,
}

/// A YOLO image inference cached for prompt and click selection.
pub struct Encoded {
    detections: Vec<Detection>,
    prototypes: Vec<f32>,
    mask_channels: usize,
    mask_height: usize,
    mask_width: usize,
    transform: Transform,
}

/// CPU inference from a fixed-size ONNX instance-segmentation export.
pub struct Model {
    runner: Arc<TypedRunnableModel>,
    labels: Vec<String>,
}

/// Whether `dir` contains the converted model and its LightCraft manifest.
pub fn is_model_dir(dir: &Path) -> bool {
    dir.join(MODEL_FILE).is_file() && dir.join(MANIFEST_FILE).is_file()
}

/// Whether the directory looks like an incomplete or complete YOLO model installation.
pub fn is_candidate_dir(dir: &Path) -> bool {
    dir.join(MODEL_FILE).exists() || dir.join(MANIFEST_FILE).exists()
}

impl Model {
    /// Load a model exported as a fixed 640 × 640 ONNX graph, plus its label manifest.
    pub fn load(dir: &Path) -> Result<Self> {
        let manifest_path = dir.join(MANIFEST_FILE);
        let manifest_file = File::open(&manifest_path).map_err(|e| Error::Yolo(format!("{}: {e}", manifest_path.display())))?;
        if manifest_file.metadata().map_err(|e| Error::Yolo(format!("{}: {e}", manifest_path.display())))?.len() > MAX_MANIFEST_BYTES {
            return Err(Error::Yolo("model manifest is too large".into()));
        }
        let mut manifest_bytes = Vec::new();
        manifest_file
            .take(MAX_MANIFEST_BYTES + 1)
            .read_to_end(&mut manifest_bytes)
            .map_err(|e| Error::Yolo(format!("{}: {e}", manifest_path.display())))?;
        if manifest_bytes.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(Error::Yolo("model manifest is too large".into()));
        }
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes).map_err(|e| Error::Yolo(format!("{}: {e}", manifest_path.display())))?;
        validate_manifest(&manifest)?;

        let model_path = dir.join(MODEL_FILE);
        let model_file = File::open(&model_path).map_err(|e| Error::Yolo(format!("{}: {e}", model_path.display())))?;
        let model_bytes = model_file.metadata().map_err(|e| Error::Yolo(format!("{}: {e}", model_path.display())))?.len();
        if model_bytes == 0 || model_bytes > MAX_MODEL_BYTES {
            return Err(Error::Yolo(format!("ONNX model size must be between 1 byte and {MAX_MODEL_BYTES} bytes")));
        }

        // Reading without a path resolver rejects external tensor files and prevents an ONNX
        // graph from making the runtime open files outside this model directory.
        let mut bounded_model_file = model_file.take(MAX_MODEL_BYTES + 1);
        let model = onnx()
            .model_for_read(&mut bounded_model_file)
            .map_err(|e| Error::Yolo(format!("invalid ONNX model: {e}")))?
            .into_optimized()
            .map_err(|e| Error::Yolo(format!("unsupported ONNX model: {e}")))?;
        let inputs = model.input_outlets().map_err(|e| Error::Yolo(format!("could not read ONNX inputs: {e}")))?;
        let outputs = model.output_outlets().map_err(|e| Error::Yolo(format!("could not read ONNX outputs: {e}")))?;
        if inputs.len() != 1 || outputs.len() != 2 {
            return Err(Error::Yolo("expected one image input and two segmentation outputs".into()));
        }
        let input_shape = model
            .input_fact(0)
            .map_err(|e| Error::Yolo(format!("could not read ONNX input shape: {e}")))?
            .shape
            .as_concrete()
            .ok_or_else(|| Error::Yolo("ONNX input dimensions must be fixed".into()))?;
        if input_shape != [1, 3, INPUT_EDGE, INPUT_EDGE] {
            return Err(Error::Yolo(format!("expected an ONNX input of [1, 3, {INPUT_EDGE}, {INPUT_EDGE}], got {input_shape:?}")));
        }
        for output in 0..outputs.len() {
            let shape = model
                .output_fact(output)
                .map_err(|e| Error::Yolo(format!("could not read ONNX output shape: {e}")))?
                .shape
                .as_concrete()
                .ok_or_else(|| Error::Yolo("ONNX output dimensions must be fixed".into()))?;
            let count = shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d)).ok_or_else(|| Error::Yolo("ONNX output shape overflows".into()))?;
            if count > MAX_OUTPUT_VALUES {
                return Err(Error::Yolo("ONNX output is too large".into()));
            }
        }
        let runner = model.into_runnable().map_err(|e| Error::Yolo(format!("could not prepare ONNX model: {e}")))?;
        Ok(Self { runner, labels: manifest.labels })
    }

    /// Run one image through the ONNX graph and cache its detections and mask prototypes.
    pub fn encode(&self, rgb: &[u8], width: usize, height: usize) -> Result<Encoded> {
        let (input, transform) = preprocess(rgb, width, height)?;
        let tensor =
            Tensor::from_shape(&[1, 3, INPUT_EDGE, INPUT_EDGE], &input).map_err(|e| Error::Yolo(format!("could not prepare the image: {e}")))?;
        let outputs = self.runner.run(tvec!(tensor.into_tvalue())).map_err(|e| Error::Yolo(format!("YOLO inference failed: {e}")))?;
        if outputs.len() != 2 {
            return Err(Error::Yolo(format!("expected two segmentation outputs, got {}", outputs.len())));
        }
        let mut parsed = Vec::with_capacity(outputs.len());
        for value in outputs {
            let shape = value.shape().to_vec();
            let count = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).ok_or_else(|| Error::Yolo("ONNX output shape overflows".into()))?;
            if count > MAX_OUTPUT_VALUES {
                return Err(Error::Yolo("ONNX output is too large".into()));
            }
            let plain = value.as_plain_ram().ok_or_else(|| Error::Yolo("ONNX output is not in CPU memory".into()))?;
            let values = plain.as_slice::<f32>().map_err(|e| Error::Yolo(format!("ONNX output must be float32: {e}")))?.to_vec();
            if values.len() != count || values.iter().any(|v| !v.is_finite()) {
                return Err(Error::Yolo("ONNX output has an invalid shape or non-finite values".into()));
            }
            parsed.push(Output { shape, values });
        }
        let (detections, prototypes, mask_channels, mask_height, mask_width) = decode_outputs(&parsed, &self.labels)?;
        Ok(Encoded { detections, prototypes, mask_channels, mask_height, mask_width, transform })
    }

    /// Select detected instance masks that contain the positive clicks and not the excluded clicks.
    pub fn segment_clicks(&self, encoded: &Encoded, clicks: &[Click]) -> Result<Option<Prediction>> {
        select_clicks(encoded, clicks)
    }

    /// Merge masks whose fixed class labels match any phrase in `text`.
    pub fn segment_text(&self, encoded: &Encoded, text: &str) -> Result<Option<Prediction>> {
        segment_text_for(&self.labels, encoded, text)
    }

    /// Merge masks whose fixed class labels match focus.
    pub fn segment_subject(&self, encoded: &Encoded, focus: Option<(f32, f32)>) -> Result<Option<Prediction>> {
        segment_subject_for(encoded, focus)
    }
}

fn select_clicks(encoded: &Encoded, clicks: &[Click]) -> Result<Option<Prediction>> {
    if clicks.is_empty() || !clicks.iter().any(|click| click.positive) {
        return Err(Error::Yolo("click an object to include it first".into()));
    }
    if clicks.iter().any(|click| !click.x.is_finite() || !click.y.is_finite() || !(0.0..=1.0).contains(&click.x) || !(0.0..=1.0).contains(&click.y)) {
        return Err(Error::Yolo("click coordinates must be finite and normalized".into()));
    }

    let mut selected = HashSet::new();
    for click in clicks.iter().filter(|click| click.positive) {
        if let Some((index, _)) = encoded
            .detections
            .iter()
            .enumerate()
            .filter(|(_, detection)| mask_logit_at(encoded, detection, click.x, click.y) > 0.0)
            .max_by(|(_, a), (_, b)| a.score.total_cmp(&b.score))
        {
            selected.insert(index);
        }
    }
    selected.retain(|&index| {
        let Some(detection) = encoded.detections.get(index) else { return false };
        !clicks.iter().filter(|click| !click.positive).any(|click| mask_logit_at(encoded, detection, click.x, click.y) > 0.0)
    });
    prediction_for(encoded, selected.into_iter().filter_map(|index| encoded.detections.get(index)))
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    if manifest.format != SCHEMA || manifest.version != 1 || manifest.image_size != INPUT_EDGE {
        return Err(Error::Yolo(format!("unsupported model manifest (expected {SCHEMA} v1 at {INPUT_EDGE} px)")));
    }
    if manifest.labels.is_empty() || manifest.labels.len() > MAX_LABELS {
        return Err(Error::Yolo(format!("model must contain between 1 and {MAX_LABELS} labels")));
    }
    let mut seen = HashSet::new();
    for label in &manifest.labels {
        let normalized = label.trim().to_lowercase();
        if normalized.is_empty() || normalized.len() > 256 || normalized.chars().any(char::is_control) || !seen.insert(normalized) {
            return Err(Error::Yolo("model labels must be unique, non-empty text of at most 256 bytes".into()));
        }
    }
    Ok(())
}

fn preprocess(rgb: &[u8], width: usize, height: usize) -> Result<(Vec<f32>, Transform)> {
    let source_len = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .filter(|&n| n <= MAX_INPUT_BYTES)
        .ok_or_else(|| Error::Yolo("image dimensions are zero, overflowed, or too large".into()))?;
    if width == 0 || height == 0 || rgb.len() != source_len {
        return Err(Error::Yolo("RGB buffer does not match image dimensions".into()));
    }
    let scale = (INPUT_EDGE as f64 / width as f64).min(INPUT_EDGE as f64 / height as f64);
    let resized_width = ((width as f64 * scale).round() as usize).clamp(1, INPUT_EDGE);
    let resized_height = ((height as f64 * scale).round() as usize).clamp(1, INPUT_EDGE);
    let pad_x = (INPUT_EDGE - resized_width) / 2;
    let pad_y = (INPUT_EDGE - resized_height) / 2;
    let mut input = vec![114.0 / 255.0; 3 * INPUT_EDGE * INPUT_EDGE];
    for y in 0..resized_height {
        let source_y = ((y as f64 + 0.5) * height as f64 / resized_height as f64 - 0.5).clamp(0.0, height.saturating_sub(1) as f64);
        let y0 = source_y.floor() as usize;
        let y1 = (y0 + 1).min(height - 1);
        let fy = (source_y - y0 as f64) as f32;
        for x in 0..resized_width {
            let source_x = ((x as f64 + 0.5) * width as f64 / resized_width as f64 - 0.5).clamp(0.0, width.saturating_sub(1) as f64);
            let x0 = source_x.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let fx = (source_x - x0 as f64) as f32;
            for channel in 0..3 {
                let at = |sx: usize, sy: usize| {
                    sy.checked_mul(width)
                        .and_then(|n| n.checked_add(sx))
                        .and_then(|n| n.checked_mul(3))
                        .and_then(|n| n.checked_add(channel))
                        .and_then(|n| rgb.get(n))
                        .copied()
                        .map(f32::from)
                        .map(|v| v / 255.0)
                };
                let pixel = |sx, sy| at(sx, sy).ok_or_else(|| Error::Yolo("RGB image is truncated".into()));
                let top = pixel(x0, y0)? * (1.0 - fx) + pixel(x1, y0)? * fx;
                let bottom = pixel(x0, y1)? * (1.0 - fx) + pixel(x1, y1)? * fx;
                let index = channel * INPUT_EDGE * INPUT_EDGE + (y + pad_y) * INPUT_EDGE + x + pad_x;
                if let Some(pixel) = input.get_mut(index) {
                    *pixel = top * (1.0 - fy) + bottom * fy;
                }
            }
        }
    }
    Ok((
        input,
        Transform {
            source_width: width,
            source_height: height,
            scale_x: resized_width as f32 / width as f32,
            scale_y: resized_height as f32 / height as f32,
            pad_x,
            pad_y,
        },
    ))
}

fn decode_outputs(outputs: &[Output], labels: &[String]) -> Result<(Vec<Detection>, Vec<f32>, usize, usize, usize)> {
    let proto_index = outputs.iter().position(|output| output.shape.len() == 4);
    let pred_index = outputs.iter().position(|output| output.shape.len() == 3);
    let (Some(proto_index), Some(pred_index)) = (proto_index, pred_index) else {
        return Err(Error::Yolo("expected rank-3 detections and rank-4 mask prototypes".into()));
    };
    if proto_index == pred_index {
        return Err(Error::Yolo("ONNX segmentation outputs are ambiguous".into()));
    }
    let proto = outputs.get(proto_index).ok_or_else(|| Error::Yolo("missing mask prototypes".into()))?;
    let [1, mask_channels, mask_height, mask_width] = proto.shape.as_slice() else {
        return Err(Error::Yolo(format!("invalid mask prototype shape: {:?}", proto.shape)));
    };
    if *mask_channels == 0 || *mask_channels > 128 || *mask_height == 0 || *mask_height > 1024 || *mask_width == 0 || *mask_width > 1024 {
        return Err(Error::Yolo("mask prototype dimensions are outside the supported limits".into()));
    }
    let Some(mask_values) = mask_channels.checked_mul(*mask_height).and_then(|n| n.checked_mul(*mask_width)) else {
        return Err(Error::Yolo("mask prototype shape overflows".into()));
    };
    if proto.values.len() != mask_values {
        return Err(Error::Yolo("mask prototype data does not match its shape".into()));
    }

    let pred = outputs.get(pred_index).ok_or_else(|| Error::Yolo("missing detections".into()))?;
    let prototypes = Prototypes { values: proto.values.clone(), channels: *mask_channels, height: *mask_height, width: *mask_width };
    let detections = match pred.shape.as_slice() {
        [1, candidates, features] if *features == 6 + *mask_channels => decode_end_to_end(pred, *candidates, &prototypes, labels.len())?,
        [1, channels, candidates] if *channels == 4 + labels.len() + *mask_channels => {
            decode_raw(pred, *candidates, RawLayout::ChannelsFirst, &prototypes, labels.len())?
        }
        [1, candidates, channels] if *channels == 4 + labels.len() + *mask_channels => {
            decode_raw(pred, *candidates, RawLayout::ChannelsLast, &prototypes, labels.len())?
        }
        _ => {
            return Err(Error::Yolo(format!(
                "unsupported YOLO segmentation output: detections {:?}, prototypes {:?}, {} labels",
                pred.shape,
                proto.shape,
                labels.len()
            )));
        }
    };
    Ok((detections, prototypes.values, prototypes.channels, prototypes.height, prototypes.width))
}

fn decode_end_to_end(pred: &Output, candidates: usize, prototypes: &Prototypes, labels: usize) -> Result<Vec<Detection>> {
    if candidates > 100_000 {
        return Err(Error::Yolo("too many ONNX detections".into()));
    }
    let features = 6 + prototypes.channels;
    let mut detections = Vec::new();
    for row in pred.values.chunks_exact(features).take(candidates) {
        let score = row.get(4).copied().unwrap_or(0.0);
        let class = row.get(5).copied().unwrap_or(-1.0);
        if score < CONFIDENCE || class < 0.0 || class >= labels as f32 || (class.fract()).abs() > 0.01 {
            continue;
        }
        let bounds = [
            row.first().copied().unwrap_or(0.0),
            row.get(1).copied().unwrap_or(0.0),
            row.get(2).copied().unwrap_or(0.0),
            row.get(3).copied().unwrap_or(0.0),
        ];
        let Some(bounds) = valid_bounds(bounds, INPUT_EDGE) else {
            continue;
        };
        let coefficients = row.get(6..features).ok_or_else(|| Error::Yolo("detection is missing mask coefficients".into()))?.to_vec();
        detections.push(Detection { label: class as usize, score, bounds, coefficients });
    }
    Ok(finish_detections(detections, false))
}

fn decode_raw(pred: &Output, candidates: usize, layout: RawLayout, prototypes: &Prototypes, labels: usize) -> Result<Vec<Detection>> {
    if candidates > 100_000 {
        return Err(Error::Yolo("too many ONNX detections".into()));
    }
    let channels = 4 + labels + prototypes.channels;
    let mut detections = Vec::new();
    for candidate in 0..candidates {
        let value = |channel: usize| -> Option<f32> {
            let index = if matches!(layout, RawLayout::ChannelsFirst) {
                channel.checked_mul(candidates)?.checked_add(candidate)?
            } else {
                candidate.checked_mul(channels)?.checked_add(channel)?
            };
            pred.values.get(index).copied()
        };
        let mut best = (0usize, CONFIDENCE);
        for label in 0..labels {
            let score = value(4 + label).ok_or_else(|| Error::Yolo("detection output is truncated".into()))?;
            if score > best.1 {
                best = (label, score);
            }
        }
        if best.1 < CONFIDENCE {
            continue;
        }
        let (cx, cy, width, height) = (
            value(0).ok_or_else(|| Error::Yolo("detection output is truncated".into()))?,
            value(1).ok_or_else(|| Error::Yolo("detection output is truncated".into()))?,
            value(2).ok_or_else(|| Error::Yolo("detection output is truncated".into()))?,
            value(3).ok_or_else(|| Error::Yolo("detection output is truncated".into()))?,
        );
        let bounds = [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0];
        let Some(bounds) = valid_bounds(bounds, INPUT_EDGE) else {
            continue;
        };
        let mut coefficients = Vec::with_capacity(prototypes.channels);
        for channel in 0..prototypes.channels {
            coefficients.push(value(4 + labels + channel).ok_or_else(|| Error::Yolo("mask coefficients are truncated".into()))?);
        }
        detections.push(Detection { label: best.0, score: best.1, bounds, coefficients });
    }
    Ok(finish_detections(detections, true))
}

fn finish_detections(mut detections: Vec<Detection>, nms: bool) -> Vec<Detection> {
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));
    if nms {
        let mut kept: Vec<Detection> = Vec::new();
        for candidate in detections {
            if kept.iter().any(|prior| prior.label == candidate.label && iou(prior.bounds, candidate.bounds) > NMS_IOU) {
                continue;
            }
            kept.push(candidate);
            if kept.len() == MAX_DETECTIONS {
                break;
            }
        }
        kept
    } else {
        detections.truncate(MAX_DETECTIONS);
        detections
    }
}

fn valid_bounds(mut bounds: [f32; 4], edge: usize) -> Option<[f32; 4]> {
    if bounds.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let edge = edge as f32;
    for coordinate in &mut bounds {
        *coordinate = coordinate.clamp(0.0, edge);
    }
    (bounds[2] > bounds[0] && bounds[3] > bounds[1]).then_some(bounds)
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let height = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = width * height;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - intersection;
    if union > 0.0 { intersection / union } else { 0.0 }
}

fn phrase_contains_label(phrase: &[String], label: &str) -> bool {
    let label: Vec<String> = label.split(|c: char| !c.is_alphanumeric()).filter(|word| !word.is_empty()).map(str::to_lowercase).collect();
    !label.is_empty() && phrase.windows(label.len()).any(|words| words == label)
}

fn segment_text_for(labels: &[String], encoded: &Encoded, text: &str) -> Result<Option<Prediction>> {
    let phrases: Vec<Vec<String>> = text
        .split([',', ';'])
        .map(|phrase| phrase.split(|c: char| !c.is_alphanumeric()).filter(|word| !word.is_empty()).map(str::to_lowercase).collect())
        .filter(|words: &Vec<String>| !words.is_empty())
        .collect();
    if phrases.is_empty() {
        return Err(Error::Yolo("describe a class label to select".into()));
    }
    let matched = encoded
        .detections
        .iter()
        .filter(|detection| labels.get(detection.label).is_some_and(|label| phrases.iter().any(|phrase| phrase_contains_label(phrase, label))));
    prediction_for(encoded, matched)
}

fn segment_subject_for(encoded: &Encoded, focus: Option<(f32, f32)>) -> Result<Option<Prediction>> {
    let matched = encoded
        .detections
        .iter()
        .filter(|detection| focus.is_none() || focus.map_or(false, |(x, y)| mask_logit_at(encoded, detection, x, y) > 0.0));
    prediction_for(encoded, matched)
}

fn prediction_for<'a>(encoded: &Encoded, detections: impl Iterator<Item = &'a Detection>) -> Result<Option<Prediction>> {
    let mut logits = vec![OUTSIDE_LOGIT; MASK_SIDE * MASK_SIDE];
    let mut score: f32 = 0.0;
    let mut found = false;
    for detection in detections {
        found = true;
        score = score.max(detection.score);
        let mask = mask_logits(encoded, detection)?;
        for (out, value) in logits.iter_mut().zip(mask) {
            *out = out.max(value);
        }
    }
    Ok(found.then_some(Prediction { logits, score }))
}

fn mask_logits(encoded: &Encoded, detection: &Detection) -> Result<Vec<f32>> {
    let proto_len = encoded.mask_height.checked_mul(encoded.mask_width).ok_or_else(|| Error::Yolo("mask prototype dimensions overflow".into()))?;
    let mut low_res = vec![0.0f32; proto_len];
    for y in 0..encoded.mask_height {
        for x in 0..encoded.mask_width {
            let mut value = 0.0;
            for channel in 0..encoded.mask_channels {
                let index = channel
                    .checked_mul(proto_len)
                    .and_then(|n| n.checked_add(y.checked_mul(encoded.mask_width)?))
                    .and_then(|n| n.checked_add(x))
                    .ok_or_else(|| Error::Yolo("mask prototype offset overflow".into()))?;
                let proto = encoded.prototypes.get(index).copied().ok_or_else(|| Error::Yolo("mask prototype data is truncated".into()))?;
                let coefficient =
                    detection.coefficients.get(channel).copied().ok_or_else(|| Error::Yolo("mask coefficients are truncated".into()))?;
                value += proto * coefficient;
            }
            low_res[y * encoded.mask_width + x] = value.clamp(-16.0, 16.0);
        }
    }
    let mut logits = vec![OUTSIDE_LOGIT; MASK_SIDE * MASK_SIDE];
    for y in 0..MASK_SIDE {
        let source_y = y as f32 / MASK_SIDE as f32 * encoded.transform.source_height as f32;
        let model_y = source_y * encoded.transform.scale_y + encoded.transform.pad_y as f32;
        for x in 0..MASK_SIDE {
            let source_x = x as f32 / MASK_SIDE as f32 * encoded.transform.source_width as f32;
            let model_x = source_x * encoded.transform.scale_x + encoded.transform.pad_x as f32;
            if model_x < detection.bounds[0] || model_y < detection.bounds[1] || model_x >= detection.bounds[2] || model_y >= detection.bounds[3] {
                continue;
            }
            let px = model_x / INPUT_EDGE as f32 * encoded.mask_width as f32 - 0.5;
            let py = model_y / INPUT_EDGE as f32 * encoded.mask_height as f32 - 0.5;
            if let Some(logit) = sample(&low_res, encoded.mask_width, encoded.mask_height, px, py)
                && let Some(out) = logits.get_mut(y * MASK_SIDE + x)
            {
                *out = logit.clamp(-16.0, 16.0);
            }
        }
    }
    Ok(logits)
}

fn mask_logit_at(encoded: &Encoded, detection: &Detection, x: f32, y: f32) -> f32 {
    let model_x = x * encoded.transform.source_width as f32 * encoded.transform.scale_x + encoded.transform.pad_x as f32;
    let model_y = y * encoded.transform.source_height as f32 * encoded.transform.scale_y + encoded.transform.pad_y as f32;
    if model_x < detection.bounds[0] || model_y < detection.bounds[1] || model_x >= detection.bounds[2] || model_y >= detection.bounds[3] {
        return OUTSIDE_LOGIT;
    }
    let proto_len = encoded.mask_height.saturating_mul(encoded.mask_width);
    let px = model_x / INPUT_EDGE as f32 * encoded.mask_width as f32 - 0.5;
    let py = model_y / INPUT_EDGE as f32 * encoded.mask_height as f32 - 0.5;
    let mut logit = 0.0;
    for channel in 0..encoded.mask_channels {
        let Some(start) = channel.checked_mul(proto_len) else {
            return OUTSIDE_LOGIT;
        };
        let Some(end) = start.checked_add(proto_len) else {
            return OUTSIDE_LOGIT;
        };
        let Some(plane) = encoded.prototypes.get(start..end) else {
            return OUTSIDE_LOGIT;
        };
        let Some(value) = sample(plane, encoded.mask_width, encoded.mask_height, px, py) else {
            return OUTSIDE_LOGIT;
        };
        let Some(coefficient) = detection.coefficients.get(channel) else {
            return OUTSIDE_LOGIT;
        };
        logit += value * coefficient;
    }
    logit.clamp(-16.0, 16.0)
}

fn sample(data: &[f32], width: usize, height: usize, x: f32, y: f32) -> Option<f32> {
    if width == 0 || height == 0 || !x.is_finite() || !y.is_finite() {
        return None;
    }
    let x = x.clamp(0.0, width.saturating_sub(1) as f32);
    let y = y.clamp(0.0, height.saturating_sub(1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let at = |sx: usize, sy: usize| data.get(sy.checked_mul(width)?.checked_add(sx)?).copied();
    let top = at(x0, y0)? * (1.0 - fx) + at(x1, y0)? * fx;
    let bottom = at(x0, y1)? * (1.0 - fx) + at(x1, y1)? * fx;
    Some(top * (1.0 - fy) + bottom * fy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Vec<String> {
        ["person", "fire hydrant"].into_iter().map(str::to_string).collect()
    }

    fn encoded() -> Encoded {
        let mut prototypes = vec![0.0; 4 * 4];
        prototypes[5] = 8.0;
        Encoded {
            detections: vec![Detection { label: 1, score: 0.9, bounds: [0.0, 0.0, 640.0, 640.0], coefficients: vec![1.0] }],
            prototypes,
            mask_channels: 1,
            mask_height: 4,
            mask_width: 4,
            transform: Transform { source_width: 640, source_height: 640, scale_x: 1.0, scale_y: 1.0, pad_x: 0, pad_y: 0 },
        }
    }

    fn output(shape: Vec<usize>, values: Vec<f32>) -> Output {
        Output { shape, values }
    }

    #[test]
    fn preprocess_letterboxes_and_rejects_bad_rgb_buffers() {
        let (input, transform) = preprocess(&[255, 0, 0, 0, 255, 0], 2, 1).unwrap();
        assert_eq!(input.len(), 3 * INPUT_EDGE * INPUT_EDGE);
        assert_eq!((transform.pad_x, transform.pad_y), (0, 160));
        assert_eq!((transform.source_width, transform.source_height), (2, 1));
        assert!(preprocess(&[0, 1, 2], 2, 1).is_err());
    }

    #[test]
    fn decodes_yolo26_end_to_end_masks_and_filters_invalid_rows() {
        let pred = output(vec![1, 2, 7], vec![10.0, 20.0, 30.0, 40.0, 0.95, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]);
        let proto = output(vec![1, 1, 1, 1], vec![2.0]);
        let (detections, _, channels, height, width) = decode_outputs(&[pred, proto], &labels()).unwrap();
        assert_eq!((channels, height, width), (1, 1, 1));
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].label, 1);
        assert_eq!(detections[0].bounds, [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(detections[0].coefficients, [1.0]);
    }

    #[test]
    fn decodes_yolo26_raw_predictions_and_keeps_distinct_classes_after_nms() {
        let mut values = vec![0.0; (4 + 2 + 1) * 3];
        let set = |values: &mut [f32], channel: usize, candidate: usize, value: f32| {
            values[channel * 3 + candidate] = value;
        };
        for candidate in 0..3 {
            set(&mut values, 0, candidate, 20.0 + candidate as f32);
            set(&mut values, 1, candidate, 20.0);
            set(&mut values, 2, candidate, 20.0);
            set(&mut values, 3, candidate, 20.0);
            set(&mut values, 6, candidate, 1.0);
        }
        set(&mut values, 4, 0, 0.95);
        set(&mut values, 5, 0, 0.1);
        set(&mut values, 4, 1, 0.8);
        set(&mut values, 5, 1, 0.05);
        set(&mut values, 4, 2, 0.1);
        set(&mut values, 5, 2, 0.99);

        let pred = output(vec![1, 7, 3], values);
        let proto = output(vec![1, 1, 1, 1], vec![1.0]);
        let (detections, _, _, _, _) = decode_outputs(&[pred, proto], &labels()).unwrap();
        assert_eq!(detections.len(), 2);
        assert_eq!(detections.iter().map(|detection| detection.label).collect::<Vec<_>>(), [1, 0]);
    }

    #[test]
    fn decodes_yoloe_raw_outputs_and_suppresses_duplicate_boxes() {
        let mut values = vec![0.0; (4 + 2 + 1) * 3];
        let set = |values: &mut [f32], channel: usize, candidate: usize, value: f32| {
            values[channel * 3 + candidate] = value;
        };
        for candidate in 0..3 {
            set(&mut values, 0, candidate, 20.0 + candidate as f32);
            set(&mut values, 1, candidate, 20.0);
            set(&mut values, 2, candidate, 20.0);
            set(&mut values, 3, candidate, 20.0);
            set(&mut values, 4, candidate, if candidate == 2 { 0.0 } else { 0.8 - candidate as f32 * 0.1 });
            set(&mut values, 5, candidate, if candidate == 2 { 0.0 } else { 0.1 });
            set(&mut values, 6, candidate, 1.0);
        }
        let pred = output(vec![1, 7, 3], values);
        let proto = output(vec![1, 1, 1, 1], vec![1.0]);
        let (detections, _, _, _, _) = decode_outputs(&[proto, pred], &labels()).unwrap();
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].label, 0);
    }

    #[test]
    fn text_matches_multiword_fixed_labels_and_clicks_reuse_cached_masks() {
        let encoded = encoded();
        let text = segment_text_for(&labels(), &encoded, "a fire hydrant, road").unwrap().unwrap();
        assert!(text.logits.iter().any(|v| *v > 0.0));
        assert!(segment_text_for(&labels(), &encoded, "street").unwrap().is_none());
        let clicks = [Click { x: 0.5, y: 0.5, positive: true }];
        assert!(select_clicks(&encoded, &clicks).unwrap().is_some());
        assert!(select_clicks(&encoded, &[Click { x: f32::NAN, y: 0.5, positive: true }]).is_err());
    }
}
