# YOLO segmentation in LightCraft

This page explains the YOLO alternative to SAM 3 for the Object and Describe tools. For the
full picture — the SAM 3 model, remote inference, the masking panel and the command API — see
[ai-masks.md](ai-masks.md).

## What YOLO and YOLOE are

**YOLO26** (You Only Look Once, generation 26) is a real-time object-detection and
instance-segmentation family from [Ultralytics](https://github.com/ultralytics/ultralytics).
It runs a single forward pass through a convolutional backbone and predicts bounding boxes,
class scores and per-instance segmentation masks in one shot. Each checkpoint is trained on a
fixed label vocabulary (COCO's 80 classes by default) baked into the model weights; Describe
matches your query against those fixed class names.

**YOLOE** (YOLO with Embeddings) extends the YOLO architecture with open-vocabulary
capability. Instead of a fixed label list, the segmentation checkpoint accepts a set of text
prompts selected at export time and bakes them into the ONNX graph as language embeddings.
This lets you describe objects in natural language — `white dog`, `person in a red jacket`,
`school bus` — without retraining, using the same ONNX runtime as YOLO26.

### How they compare to SAM 3

| | YOLO26 | YOLOE | SAM 3 |
|---|---|---|---|
| Describe prompts | fixed class names (e.g. COCO 80) | prompts you choose at export time | any free-form concept at query time |
| Object click | ✅ picks the highest-scoring instance under the click | ✅ same | ✅ full prompt-based segmentation |
| GPU required | ✗ CPU-only | ✗ CPU-only | ✅ GPU preferred (CPU fallback) |
| Weights included | ✗ you convert offline | ✗ you convert offline | ✗ optional download |
| Detail refinement pass | ✗ | ✗ | ✅ |
| Guided-filter edge refinement | ✅ | ✅ | ✅ |

YOLO models are lighter and start instantly. SAM 3 supports any concept without re-exporting
and delivers finer edges through its detail pass. Use YOLO when you want faster, offline-only
inference with a known set of labels; use SAM 3 when you need open-ended prompts or maximum
edge quality.

## Where to get a checkpoint

LightCraft **does not download, include, or redistribute** YOLO weights. Convert a checkpoint
you trust and keep it under its own licence.

Download an **instance-segmentation** checkpoint (the filename must include `-seg`):

- **YOLO26-seg** — <https://github.com/ultralytics/assets/releases> (look for
  `yolo26n-seg.pt`, `yolo26s-seg.pt`, `yolo26m-seg.pt`, `yolo26l-seg.pt`, `yolo26x-seg.pt`
  — nano to extra-large; larger = more accurate, slower).
- **YOLOE-seg** — <https://github.com/ultralytics/assets/releases> (look for
  `yoloe-11n-seg.pt`, `yoloe-11s-seg.pt`, `yoloe-11m-seg.pt`, `yoloe-11l-seg.pt`,
  `yoloe-11x-seg.pt`).

The `l` (large) segmentation models are recommended for both YOLO26 and YOLOE.

> [!IMPORTANT]
> Only segmentation checkpoints work. Detection-only checkpoints (no `-seg` in the name) are
> rejected by the exporter and by LightCraft's runtime.

The checkpoint licences are the responsibility of whoever downloads and uses them. Keep the
`.pt` file and the exported weights in a location that complies with the model licence.

## Converting a checkpoint with `tools/export-yolo-seg.py`

LightCraft's runtime does not load `.pt` files directly. The exporter converts a trusted
checkpoint to an ONNX graph plus a small JSON manifest, the two files LightCraft actually
loads.

> [!CAUTION]
> Loading a `.pt` checkpoint executes Python pickle code. Only convert files you downloaded
> from a trusted source. The script prints a warning before loading.

### 1. Create a conversion environment

```sh
python3 -m venv .venv-yolo-seg
.venv-yolo-seg/bin/pip install -r tools/requirements-export-yolo-seg.txt
```

This installs `ultralytics[export]` (Ultralytics and its ONNX export dependencies) in an
isolated virtual environment. Keep this environment separate from any other Python project.

### 2. Export a YOLO26 checkpoint

```sh
.venv-yolo-seg/bin/python tools/export-yolo-seg.py \
  --input-pt "$HOME/Downloads/yolo26/yolo26l-seg.pt" \
  --output  "$HOME/Downloads/lightcraft-yolo26" \
  --zip
```

The script:

1. Loads the checkpoint (shows the pickle warning).
2. Checks that the task is `segment`; exits with an error otherwise.
3. Reads the model's label vocabulary (the COCO 80 classes by default).
4. Exports a fixed-size ONNX graph (`imgsz × imgsz`, default 640 × 640) in a temporary
   staging area.
5. Writes `model.onnx` and `lightcraft-yolo.json` into a folder named after the checkpoint
   stem (`yolo26l-seg/`) inside `--output`.
6. With `--zip`, also creates `yolo26l-seg.zip` in `--output` — a ZIP containing the named
   model folder.

The source checkpoint is never modified and is not included in the output.

### 3. Export a YOLOE checkpoint

YOLOE needs a list of text prompts baked in at export time. Without `--labels` the bundled
`tools/yoloe-default-labels.txt` is used (512 starter prompts covering people, animals,
vehicles, objects and clothes).

```sh
.venv-yolo-seg/bin/python tools/export-yolo-seg.py \
  --input-pt "$HOME/Downloads/yoloe-11l-seg.pt" \
  --output  "$HOME/Downloads/lightcraft-yoloe" \
  --zip
```

For custom prompts, write one phrase per line in a UTF-8 text file and pass it with `--labels`:

```sh
.venv-yolo-seg/bin/python tools/export-yolo-seg.py \
  --input-pt "$HOME/Downloads/yoloe-11l-seg.pt" \
  --labels   "$HOME/Downloads/my-prompts.txt" \
  --output  "$HOME/Downloads/lightcraft-yoloe-custom" \
  --zip
```

`my-prompts.txt` example:
```
white dog
black dog
person in a red jacket
school bus
```

> [!IMPORTANT]
> The prompts are baked into the ONNX model and manifest at export time and **cannot be
> changed afterwards**. Re-export with a different `--labels` file to change them.

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `-i` / `--input-pt` | _(required)_ | Path to the `.pt` segmentation checkpoint |
| `--kind` | `auto` | `auto` (infer from filename), `yolo26`, or `yoloe` |
| `-s` / `--imgsz` | `640` | Square input size in pixels; multiple of 32, 32–2048 |
| `--labels` | bundled `yoloe-default-labels.txt` | YOLOE only: text file with one prompt per line |
| `--output` | directory containing `--input-pt` | Parent directory for the output model folder |
| `-z` / `--zip` | off | Also create `<checkpoint-name>.zip` in `--output` |

`--kind auto` infers the type from the checkpoint filename: the name must contain `yolo26` or
`yoloe` (case-insensitive). Pass `--kind` explicitly when it cannot be inferred.

A YOLOE export with many labels at a large `--imgsz` can exceed LightCraft's
20 000 000-value ONNX output limit. The script prints a warning and suggests reducing
`--imgsz` or the number of labels.

### Output layout

```
<output>/
  <checkpoint-stem>/          ← the model folder LightCraft loads
    model.onnx
    lightcraft-yolo.json
  <checkpoint-stem>.zip       ← (with --zip) same files in a named sub-folder
```

`lightcraft-yolo.json` records the format, schema version, image size and label list:

```json
{
  "format": "lightcraft-yolo-seg",
  "version": 1,
  "image_size": 640,
  "labels": ["person", "bicycle", "car", "..."]
}
```

## Importing the model into LightCraft

### Option A — Settings ▸ AI Mask (recommended)

1. Open **Settings → AI Mask** (or press **M** to open the Masking panel, then click the
   model indicator).
2. Click **Install model…**.
3. Select one of:
   - the **model folder** (`yolo26l-seg/` or `yoloe-11l-seg/`) — both files must be present,
   - the **ZIP archive** (`yolo26l-seg.zip`) — the archive must contain the two files in a
     named sub-folder,
   - the two **individual files** (`model.onnx` and `lightcraft-yolo.json`) together.
4. ONNX validation runs in the background. When it finishes, the model appears in the list.
5. Click the model's name to make it active; LightCraft uses it for the current session and
   remembers it across restarts.

Installed models are stored under `<LightCraft config>/models/<model-id>/`:

| Platform | Config folder |
|---|---|
| macOS | `~/Library/Application Support/LightCraft/` |
| Windows | `%APPDATA%\LightCraft\` |
| Linux | `$XDG_CONFIG_HOME/lightcraft/` or `~/.config/lightcraft/` |

### Option B — Environment variable (developers, CI, agents)

Point `LIGHTCRAFT_SEGMENTATION_MODEL_DIR` at the model folder before launching LightCraft.
This takes precedence over the Settings selection and cannot be changed from Settings.

```sh
# YOLO26
LIGHTCRAFT_SEGMENTATION_MODEL_DIR="$HOME/Downloads/lightcraft-yolo26/yolo26l-seg" lightcraft

# YOLOE (default labels)
LIGHTCRAFT_SEGMENTATION_MODEL_DIR="$HOME/Downloads/lightcraft-yoloe/yoloe-11l-seg" lightcraft

# YOLOE (custom labels, convert + launch in one go)
.venv-yolo-seg/bin/python tools/export-yolo-seg.py \
  --input-pt "$HOME/Downloads/yoloe-11l-seg.pt" \
  --output   "$HOME/Downloads/lightcraft-yoloe"
LIGHTCRAFT_SEGMENTATION_MODEL_DIR="$HOME/Downloads/lightcraft-yoloe/yoloe-11l-seg" lightcraft
```

## Using YOLO masks in the Masking panel

The masking panel (M) works identically for YOLO and SAM 3. The only differences are:

- **Object** — clicks pick the highest-scoring predicted instance whose mask covers the click
  point. ⌥-click (Alt) a point to exclude the instances that cover it. Up to 64 clicks.
- **Describe** — the query is matched against the **baked-in labels** (YOLO26: fixed class
  names; YOLOE: your export-time prompts). Every instance whose label is a substring-match
  of the query joins the mask. Comma-separated queries (`car, truck`) match multiple labels.
  Free-form concepts not in the label list return nothing; re-export with the right prompts or
  switch to SAM 3.
- No **Detail** refinement pass (YOLO produces whole-image detections in one pass; there is
  no SAM-style zoom-in crop).
- No download dialog — the model is not downloaded by LightCraft; install it yourself first.

The **Edge** slider (−100…100), **+ / −** composition and guided-filter edge refinement work
the same as with SAM 3.

`segment.model.status` on the control channel / MCP reports `sizeBytes` (the ONNX file size)
and `null` for `license` / `licenseUrl` because LightCraft does not know the user's model
licence. All other mask commands (`mask.add`, `mask.objectPoint`, `mask.addComponent`, …) work
unchanged.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Exporter fails: `checkpoint must be an instance-segmentation model` | Detection-only `.pt` file | Download a `-seg` checkpoint |
| Exporter fails: `cannot infer model type from checkpoint filename` | Filename has no `yolo26` or `yoloe` | Pass `--kind yolo26` or `--kind yoloe` |
| Exporter prints output size warning | Too many labels for the chosen `--imgsz` | Use a smaller `--imgsz` (e.g. `--imgsz 320`) or fewer labels |
| Settings install fails: "ONNX model size must be between…" | Model file is 0 bytes or over 1 GB | Re-export; check disk space |
| Settings install fails: "expected one image input and two segmentation outputs" | Wrong ONNX export settings | Re-export with the script; do not export manually |
| Object clicks select nothing | No predicted instance covers the click | Try clicking a different part of the object, or switch to SAM 3 |
| Describe matches nothing | Query not in the label list | Use a label that is in the manifest's `labels` array; for YOLOE, re-export with that prompt |
| `LIGHTCRAFT_SEGMENTATION_MODEL_DIR` ignored | A different env var is set at the same time | `LIGHTCRAFT_SEGMENTATION_MODEL_DIR` takes priority over `LIGHTCRAFT_SAM3_DIR`; check that neither is accidentally empty |
