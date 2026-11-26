//! Enabled local AI mask backends, in loading priority order.

use std::path::Path;

use lightcraft_segment::{AiMaskInterface, Error};

struct Model {
    can_load: fn(&Path) -> bool,
    load: fn(&Path, bool) -> lightcraft_segment::Result<Box<dyn AiMaskInterface>>,
}

static MODELS: &[Model] = &[
    #[cfg(feature = "yolo")]
    Model { can_load: lightcraft_segment::Yolo::can_load, load: |dir, _cpu| Ok(Box::new(lightcraft_segment::Yolo::load(dir)?)) },
    #[cfg(feature = "sam")]
    Model {
        can_load: lightcraft_segment::Sam3::can_load,
        load: |dir, cpu| {
            let model = if cpu { lightcraft_segment::Sam3::load_cpu(dir)? } else { lightcraft_segment::Sam3::load(dir)? };
            Ok(Box::new(model))
        },
    },
];

pub(super) const AVAILABLE: bool = !MODELS.is_empty();

fn find(dir: &Path) -> Option<&'static Model> {
    MODELS.iter().find(|model| (model.can_load)(dir))
}

pub(super) fn can_load(dir: &Path) -> bool {
    find(dir).is_some()
}

pub(super) fn load(dir: &Path, cpu: bool) -> lightcraft_segment::Result<Box<dyn AiMaskInterface>> {
    let model = find(dir).ok_or_else(|| Error::Missing(dir.to_path_buf()))?;
    (model.load)(dir, cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn detection_and_loading_use_only_enabled_backends() {
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("lc-mask-backends-{}-{nonce}", std::process::id()));
        let sam = root.join("sam");
        let yolo = root.join("yolo");
        fs::create_dir_all(&sam).unwrap();
        fs::create_dir_all(&yolo).unwrap();
        for name in ["model.safetensors", "vocab.json", "merges.txt"] {
            fs::write(sam.join(name), b"invalid").unwrap();
        }
        for name in ["model.onnx", "lightcraft-yolo.json"] {
            fs::write(yolo.join(name), b"invalid").unwrap();
        }

        for (dir, enabled) in [(&sam, cfg!(feature = "sam")), (&yolo, cfg!(feature = "yolo"))] {
            assert_eq!(can_load(dir), enabled);
            assert_eq!(super::super::Segmenter { dir: Some(dir.clone()), ..Default::default() }.installed(), enabled);
            let error = match load(dir, true) {
                Ok(_) => panic!("invalid test model unexpectedly loaded"),
                Err(error) => error,
            };
            assert_eq!(matches!(error, Error::Missing(_)), !enabled, "{error}");
        }
        assert_eq!(super::super::Segmenter::AVAILABLE, cfg!(feature = "sam") || cfg!(feature = "yolo"));
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(all(feature = "sam", feature = "yolo"))]
    #[test]
    fn yolo_takes_priority_when_both_layouts_are_present() {
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("lc-mask-priority-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        for name in ["model.safetensors", "vocab.json", "merges.txt", "model.onnx", "lightcraft-yolo.json"] {
            fs::write(dir.join(name), b"invalid").unwrap();
        }
        assert!(matches!(load(&dir, true), Err(Error::Yolo(_))));
        fs::remove_dir_all(dir).unwrap();
    }
}
