//! The common AI mask interface for segmentation models (SAM 3 and YOLO).

use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
use candle_core::Device;

use crate::{Click, Error, Prediction, Result};
#[cfg(not(target_arch = "wasm32"))]
use crate::{Sam3, Yolo, best_device};

/// Type-erased cached image features computed by [`AiMaskInterface::encode`].
pub trait EncodedImage: std::any::Any + Send {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl<T: std::any::Any + Send> EncodedImage for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// The common interface for AI masking models (SAM 3, YOLO).
pub trait AiMaskInterface: Send {
    /// Friendly name of this model backend (e.g. "SAM 3", "YOLO").
    fn name(&self) -> &'static str;

    /// Run the model backbone on an RGB image (8-bit, row-major, `width × height`).
    fn encode(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Box<dyn EncodedImage>>;

    /// Pre-warm interactive click features for `encoded` ahead of time (optional optimization).
    fn prepare_clicks(&mut self, encoded: &mut dyn EncodedImage) -> Result<()> {
        let _ = encoded;
        Ok(())
    }

    /// Predict a mask from positive and negative clicks.
    fn segment_clicks(&mut self, encoded: &mut dyn EncodedImage, clicks: &[Click]) -> Result<Option<Prediction>>;

    /// Predict a mask from a text prompt (supports comma and semicolon separated concepts).
    fn segment_text(&mut self, encoded: &mut dyn EncodedImage, text: &str) -> Result<Option<Prediction>>;

    /// Segment the photo's subject among people and animals (or at focus point).
    fn segment_subject(&mut self, encoded: &mut dyn EncodedImage, focus: Option<(f32, f32)>) -> Result<Option<Prediction>>;
}

impl<T: AiMaskInterface + ?Sized> AiMaskInterface for Box<T> {
    fn name(&self) -> &'static str {
        (**self).name()
    }

    fn encode(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Box<dyn EncodedImage>> {
        (**self).encode(rgb, width, height)
    }

    fn prepare_clicks(&mut self, encoded: &mut dyn EncodedImage) -> Result<()> {
        (**self).prepare_clicks(encoded)
    }

    fn segment_clicks(&mut self, encoded: &mut dyn EncodedImage, clicks: &[Click]) -> Result<Option<Prediction>> {
        (**self).segment_clicks(encoded, clicks)
    }

    fn segment_text(&mut self, encoded: &mut dyn EncodedImage, text: &str) -> Result<Option<Prediction>> {
        (**self).segment_text(encoded, text)
    }

    fn segment_subject(&mut self, encoded: &mut dyn EncodedImage, focus: Option<(f32, f32)>) -> Result<Option<Prediction>> {
        (**self).segment_subject(encoded, focus)
    }
}

/// Whether `dir` contains a model that can be loaded by any supported backend.
pub fn can_load(dir: &Path) -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Yolo::can_load(dir) || Sam3::can_load(dir)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = dir;
        false
    }
}

#[cfg(not(target_arch = "wasm32"))]
/// Load a segmentation model using the best available device.
pub fn load_model(dir: &Path) -> Result<Box<dyn AiMaskInterface>> {
    load_model_on(dir, best_device())
}

#[cfg(not(target_arch = "wasm32"))]
/// Load a segmentation model on the CPU.
pub fn load_model_cpu(dir: &Path) -> Result<Box<dyn AiMaskInterface>> {
    load_model_on(dir, Device::Cpu)
}

#[cfg(not(target_arch = "wasm32"))]
/// Load a segmentation model on a specific device (SAM 3 uses `device`; YOLO uses CPU).
pub fn load_model_on(dir: &Path, device: Device) -> Result<Box<dyn AiMaskInterface>> {
    if Yolo::can_load(dir) {
        Ok(Box::new(Yolo::load(dir)?))
    } else if Sam3::can_load(dir) {
        Ok(Box::new(Sam3::load_on(dir, device)?))
    } else {
        Err(Error::Missing(dir.to_path_buf()))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AiMaskInterface for Sam3 {
    fn name(&self) -> &'static str {
        "SAM 3"
    }

    fn encode(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Box<dyn EncodedImage>> {
        let enc = Sam3::encode(self, rgb, width, height)?;
        Ok(Box::new(enc))
    }

    fn prepare_clicks(&mut self, encoded: &mut dyn EncodedImage) -> Result<()> {
        let enc = encoded.as_any_mut().downcast_mut::<crate::Encoded>().ok_or_else(|| Error::Model("expected SAM 3 encoded image".into()))?;
        Sam3::prepare_clicks(self, enc)
    }

    fn segment_clicks(&mut self, encoded: &mut dyn EncodedImage, clicks: &[Click]) -> Result<Option<Prediction>> {
        let enc = encoded.as_any_mut().downcast_mut::<crate::Encoded>().ok_or_else(|| Error::Model("expected SAM 3 encoded image".into()))?;
        let prediction = Sam3::segment_clicks(self, enc, clicks)?;
        Ok(Some(prediction))
    }

    fn segment_text(&mut self, encoded: &mut dyn EncodedImage, text: &str) -> Result<Option<Prediction>> {
        let enc = encoded.as_any_mut().downcast_mut::<crate::Encoded>().ok_or_else(|| Error::Model("expected SAM 3 encoded image".into()))?;
        let mut merged_logits: Option<Vec<f32>> = None;
        let mut max_score: f32 = 0.0;
        for phrase in text.split([',', ';']).map(str::trim).filter(|p| !p.is_empty()) {
            if let Some(p) = Sam3::segment_text(self, enc, phrase, 0.5)? {
                max_score = max_score.max(p.score);
                if let Some(m) = &mut merged_logits {
                    for (a, b) in m.iter_mut().zip(p.logits) {
                        *a = a.max(b);
                    }
                } else {
                    merged_logits = Some(p.logits);
                }
            }
        }
        Ok(merged_logits.map(|logits| Prediction { logits, score: max_score }))
    }

    fn segment_subject(&mut self, encoded: &mut dyn EncodedImage, focus: Option<(f32, f32)>) -> Result<Option<Prediction>> {
        let enc = encoded.as_any_mut().downcast_mut::<crate::Encoded>().ok_or_else(|| Error::Model("expected SAM 3 encoded image".into()))?;
        Sam3::segment_subject(self, enc, focus, 0.5)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AiMaskInterface for Yolo {
    fn name(&self) -> &'static str {
        "YOLO"
    }

    fn encode(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Box<dyn EncodedImage>> {
        let enc = Yolo::encode(self, rgb, width, height)?;
        Ok(Box::new(enc))
    }

    fn prepare_clicks(&mut self, _encoded: &mut dyn EncodedImage) -> Result<()> {
        Ok(())
    }

    fn segment_clicks(&mut self, encoded: &mut dyn EncodedImage, clicks: &[Click]) -> Result<Option<Prediction>> {
        let enc = encoded.as_any_mut().downcast_mut::<crate::yolo::Encoded>().ok_or_else(|| Error::Yolo("expected YOLO encoded image".into()))?;
        Yolo::segment_clicks(self, enc, clicks)
    }

    fn segment_text(&mut self, encoded: &mut dyn EncodedImage, text: &str) -> Result<Option<Prediction>> {
        let enc = encoded.as_any_mut().downcast_mut::<crate::yolo::Encoded>().ok_or_else(|| Error::Yolo("expected YOLO encoded image".into()))?;
        Yolo::segment_text(self, enc, text)
    }

    fn segment_subject(&mut self, encoded: &mut dyn EncodedImage, focus: Option<(f32, f32)>) -> Result<Option<Prediction>> {
        let enc = encoded.as_any_mut().downcast_mut::<crate::yolo::Encoded>().ok_or_else(|| Error::Yolo("expected YOLO encoded image".into()))?;
        Yolo::segment_subject(self, enc, focus)
    }
}
