//! Managed installs for local segmentation models.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

const SAM_WEIGHTS: &str = "model.safetensors";
const YOLO_MODEL: &str = "model.onnx";
const YOLO_MANIFEST: &str = "lightcraft-yolo.json";
const MAX_SAM_FILE_BYTES: u64 = 8 << 30;
const MAX_TOTAL_BYTES: u64 = 9 << 30;
const MAX_YOLO_FILE_BYTES: u64 = 1 << 30;
const MAX_ZIP_ENTRIES: usize = 4096;
const MODEL_FILES: [&str; 5] = [SAM_WEIGHTS, "vocab.json", "merges.txt", YOLO_MODEL, YOLO_MANIFEST];
static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct Model {
    pub id: String,
    pub kind: &'static str,
    pub dir: PathBuf,
    pub size_bytes: u64,
    pub managed: bool,
}

pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 48
        && id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && id.as_bytes().first().is_some_and(u8::is_ascii_alphanumeric)
        && id.as_bytes().last().is_some_and(u8::is_ascii_alphanumeric)
}

pub fn is_supported_model(dir: &Path) -> bool {
    if is_sam_layout(dir) {
        return ["model.safetensors", "vocab.json", "merges.txt"]
            .iter()
            .all(|name| fs::metadata(dir.join(name)).is_ok_and(|m| m.is_file() && m.len() > 0 && m.len() <= MAX_SAM_FILE_BYTES));
    }
    if is_yolo_layout(dir) {
        #[cfg(feature = "sam")]
        return [YOLO_MODEL, YOLO_MANIFEST]
            .iter()
            .all(|name| fs::metadata(dir.join(name)).is_ok_and(|m| m.is_file() && m.len() > 0 && m.len() <= file_limit(name)));
        #[cfg(not(feature = "sam"))]
        return false;
    }
    false
}

pub fn list(store: &Path, active_dir: Option<&Path>) -> Result<Vec<Model>, String> {
    let mut models = Vec::new();
    match fs::read_dir(store) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry.map_err(|e| format!("could not list {}: {e}", store.display()))?;
                let name = entry.file_name().to_string_lossy().into_owned();
                if !valid_id(&name) || !entry.file_type().map_err(|e| format!("{}: {e}", entry.path().display()))?.is_dir() {
                    continue;
                }
                let dir = entry.path();
                if let Some(model) = describe(name, dir, true)? {
                    models.push(model);
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("could not list {}: {e}", store.display())),
    }
    if let Some(dir) = active_dir
        && !models.iter().any(|m| m.dir == dir)
        && is_supported_model(dir)
    {
        let id = dir.file_name().and_then(|n| n.to_str()).map(slug).filter(|s| valid_id(s)).unwrap_or_else(|| "external-model".into());
        if let Some(model) = describe(id, dir.to_path_buf(), false)? {
            models.push(model);
        }
    }
    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

fn describe(id: String, dir: PathBuf, managed: bool) -> Result<Option<Model>, String> {
    let kind = if is_sam_layout(&dir) {
        "sam3"
    } else if is_yolo_layout(&dir) {
        "yolo"
    } else {
        return Ok(None);
    };
    if !is_supported_model(&dir) {
        return Ok(None);
    }
    let names: &[&str] = if kind == "sam3" { &[SAM_WEIGHTS, "vocab.json", "merges.txt"] } else { &[YOLO_MODEL, YOLO_MANIFEST] };
    let size_bytes = names
        .iter()
        .try_fold(0_u64, |sum, name| sum.checked_add(fs::metadata(dir.join(name)).ok()?.len()))
        .ok_or_else(|| format!("could not read model size in {}", dir.display()))?;
    Ok(Some(Model { id, kind, dir, size_bytes, managed }))
}

pub fn install(store: &Path, sources: &[String], acknowledged_sam: bool) -> Result<Model, String> {
    if sources.is_empty() {
        return Err("select model files, a model folder, or a ZIP archive first".into());
    }
    if sources.len() > MODEL_FILES.len() {
        return Err(format!("select no more than {} model files at once", MODEL_FILES.len()));
    }
    let stage = create_stage(store)?;
    let mut cleanup = StageCleanup(Some(stage.clone()));
    let paths = sources.iter().map(PathBuf::from).collect::<Vec<_>>();
    if paths.len() == 1 && paths[0].is_dir() {
        copy_folder_bundle(&paths[0], &stage)?;
    } else if paths.len() == 1 && paths[0].extension().is_some_and(|e| e.eq_ignore_ascii_case("zip")) {
        extract_zip(&paths[0], &stage)?;
    } else {
        copy_selected_files(&paths, &stage)?;
    }

    let kind = validate_stage(&stage)?;
    let _ = acknowledged_sam;

    fs::create_dir_all(store).map_err(|e| format!("could not create {}: {e}", store.display()))?;
    let source_name = if kind == "sam3" { "sam3".to_string() } else { source_name(&paths) };
    let mut id = slug(&source_name);
    if !valid_id(&id) {
        id = "ai-mask-model".into();
    }
    if kind == "sam3" && store.join(&id).exists() {
        return Err(format!("SAM 3 already has a model folder at {}; remove or choose another folder first", store.join(&id).display()));
    }
    let mut suffix = 2_u32;
    let mut target = store.join(&id);
    while target.exists() {
        id = format!("{}-{suffix}", slug(&source_name));
        if !valid_id(&id) {
            id = format!("ai-mask-model-{suffix}");
        }
        target = store.join(&id);
        suffix = suffix.saturating_add(1);
        if suffix == u32::MAX {
            return Err("could not choose a unique model name".into());
        }
    }
    fs::rename(&stage, &target).map_err(|e| format!("could not install the model in {}: {e}", target.display()))?;
    cleanup.0 = None;
    describe(id, target, true)?.ok_or_else(|| "the installed model could not be read".into())
}

pub fn remove(store: &Path, id: &str) -> Result<(), String> {
    if !valid_id(id) {
        return Err("not a model id".into());
    }
    let target = store.join(id);
    let metadata = fs::symlink_metadata(&target).map_err(|e| format!("could not find model `{id}`: {e}"))?;
    if !metadata.file_type().is_dir() || !is_supported_model(&target) {
        return Err("that managed segmentation model is not installed or is invalid".into());
    }
    fs::remove_dir_all(&target).map_err(|e| format!("could not remove {}: {e}", target.display()))
}

pub fn write_active(store: &Path, id: Option<&str>) -> Result<(), String> {
    let active = store.join("active.txt");
    match id {
        Some(id) => {
            if !valid_id(id) {
                return Err("not a model id".into());
            }
            fs::create_dir_all(store).map_err(|e| format!("could not create {}: {e}", store.display()))?;
            let part = store.join("active.txt.part");
            let mut file =
                OpenOptions::new().write(true).create_new(true).open(&part).map_err(|e| format!("could not create {}: {e}", part.display()))?;
            if let Err(e) = file.write_all(id.as_bytes()).and_then(|()| file.sync_all()) {
                let _ = fs::remove_file(&part);
                return Err(format!("could not write {}: {e}", part.display()));
            }
            fs::rename(&part, &active).map_err(|e| {
                let _ = fs::remove_file(&part);
                format!("could not select model `{id}`: {e}")
            })
        }
        None => match fs::remove_file(&active) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(format!("could not clear the active model: {e}")),
        },
    }
}

fn create_stage(store: &Path) -> Result<PathBuf, String> {
    fs::create_dir_all(store).map_err(|e| format!("could not create {}: {e}", store.display()))?;
    for _ in 0..32 {
        let n = NEXT_STAGE.fetch_add(1, Ordering::Relaxed);
        let path = store.join(format!(".install-{}-{n}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("could not create model staging folder: {e}")),
        }
    }
    Err("could not create a unique model staging folder".into())
}

struct StageCleanup(Option<PathBuf>);

impl Drop for StageCleanup {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

fn copy_folder_bundle(folder: &Path, stage: &Path) -> Result<(), String> {
    let mut bundles = Vec::new();
    let direct = supported_files_in(folder)?;
    if direct.values().next().is_some() {
        bundles.push(direct);
    }
    let mut children = fs::read_dir(folder).map_err(|e| format!("could not list {}: {e}", folder.display()))?;
    for child in &mut children {
        let child = child.map_err(|e| format!("could not list {}: {e}", folder.display()))?;
        if child.file_type().map_err(|e| format!("{}: {e}", child.path().display()))?.is_dir() {
            let files = supported_files_in(&child.path())?;
            if !files.is_empty() {
                bundles.push(files);
            }
        }
    }
    if bundles.len() != 1 {
        return Err(if bundles.is_empty() {
            "the selected folder does not contain a supported SAM 3 or YOLO segmentation model".into()
        } else {
            "the selected folder contains multiple model folders; choose one model folder at a time".into()
        });
    }
    copy_file_map(&bundles[0], stage)
}

fn supported_files_in(folder: &Path) -> Result<BTreeMap<String, PathBuf>, String> {
    let entries = fs::read_dir(folder).map_err(|e| format!("could not list {}: {e}", folder.display()))?;
    let mut files = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("could not list {}: {e}", folder.display()))?;
        let file_type = entry.file_type().map_err(|e| format!("{}: {e}", entry.path().display()))?;
        if !file_type.is_file() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_string) else { continue };
        if let Some(canonical) = canonical_file_name(&name) {
            if files.insert(canonical.to_string(), entry.path()).is_some() {
                return Err(format!("{} contains duplicate `{canonical}` files", folder.display()));
            }
        }
    }
    Ok(files)
}

fn copy_selected_files(paths: &[PathBuf], stage: &Path) -> Result<(), String> {
    let mut files = BTreeMap::new();
    for path in paths {
        let name = path.file_name().and_then(|n| n.to_str()).ok_or_else(|| format!("invalid model file path: {}", path.display()))?;
        let Some(canonical) = canonical_file_name(name) else {
            return Err(format!("unsupported model file `{name}`"));
        };
        if files.insert(canonical.to_string(), path.clone()).is_some() {
            return Err(format!("more than one selected file is named `{canonical}`"));
        }
    }
    copy_file_map(&files, stage)
}

fn copy_file_map(files: &BTreeMap<String, PathBuf>, stage: &Path) -> Result<(), String> {
    let mut total = 0_u64;
    for (name, source) in files {
        let metadata = fs::metadata(source).map_err(|e| format!("could not read {}: {e}", source.display()))?;
        if !metadata.is_file() {
            return Err(format!("{} is not a regular file", source.display()));
        }
        let max = file_limit(name);
        if metadata.len() == 0 || metadata.len() > max {
            return Err(format!("{} must be between 1 byte and {max} bytes", source.display()));
        }
        total = total.checked_add(metadata.len()).ok_or("model size overflows")?;
        if total > MAX_TOTAL_BYTES {
            return Err("the model files exceed the 9 GiB installation limit".into());
        }
        let input = File::open(source).map_err(|e| format!("could not read {}: {e}", source.display()))?;
        copy_bounded(input, &stage.join(name), metadata.len(), max)?;
    }
    Ok(())
}

fn extract_zip(path: &Path, stage: &Path) -> Result<(), String> {
    let file = File::open(path).map_err(|e| format!("could not open {}: {e}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| format!("invalid ZIP archive: {e}"))?;
    if archive.len() > MAX_ZIP_ENTRIES {
        return Err(format!("ZIP archive has more than {MAX_ZIP_ENTRIES} entries"));
    }
    let mut found = BTreeMap::new();
    let mut total = 0_u64;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(|e| format!("invalid ZIP entry {index}: {e}"))?;
        let name = entry.name().replace('\\', "/");
        if name.starts_with('/') || name.split('/').any(|part| part == "..") || name.as_bytes().contains(&0) {
            return Err(format!("unsafe path in ZIP archive: {name}"));
        }
        if entry.is_dir() {
            continue;
        }
        let basename = name.rsplit('/').next().unwrap_or_default();
        let Some(canonical) = canonical_file_name(basename) else { continue };
        if found.contains_key(canonical) {
            return Err(format!("ZIP archive contains duplicate `{canonical}` files"));
        }
        let size = entry.size();
        let max = file_limit(canonical);
        if size == 0 || size > max {
            return Err(format!("ZIP member `{name}` must be between 1 byte and {max} bytes"));
        }
        total = total.checked_add(size).ok_or("model size overflows")?;
        if total > MAX_TOTAL_BYTES {
            return Err("the model files exceed the 9 GiB installation limit".into());
        }
        let target = stage.join(canonical);
        let mut output = File::create(&target).map_err(|e| format!("could not create {}: {e}", target.display()))?;
        let copied = std::io::copy(&mut entry.by_ref().take(max + 1), &mut output).map_err(|e| format!("could not extract `{name}`: {e}"))?;
        if copied != size || copied > max {
            return Err(format!("ZIP member `{name}` has an invalid size"));
        }
        found.insert(canonical.to_string(), target);
    }
    if found.is_empty() {
        return Err("ZIP archive does not contain a supported SAM 3 or YOLO segmentation model".into());
    }
    Ok(())
}

fn copy_bounded(mut input: impl Read, target: &Path, expected: u64, max: u64) -> Result<(), String> {
    let mut output = File::create(target).map_err(|e| format!("could not create {}: {e}", target.display()))?;
    let copied = std::io::copy(&mut input.by_ref().take(max + 1), &mut output).map_err(|e| format!("could not copy {}: {e}", target.display()))?;
    if copied != expected || copied > max {
        return Err(format!("{} changed size while being copied", target.display()));
    }
    output.sync_all().map_err(|e| format!("could not write {}: {e}", target.display()))
}

fn validate_stage(stage: &Path) -> Result<&'static str, String> {
    let sam = ["model.safetensors", "vocab.json", "merges.txt"].iter().all(|name| stage.join(name).is_file());
    let yolo = is_yolo_layout(stage);
    match (sam, yolo) {
        (true, false) if is_supported_model(stage) => Ok("sam3"),
        (false, true) if is_supported_model(stage) => {
            #[cfg(feature = "sam")]
            {
                lightcraft_segment::yolo::Model::load(stage).map_err(|e| e.to_string())?;
                Ok("yolo")
            }
            #[cfg(not(feature = "sam"))]
            {
                Err("this build cannot validate YOLO segmentation models".into())
            }
        }
        (true, false) => Err("the SAM 3 files are incomplete or invalid".into()),
        (false, true) => {
            #[cfg(feature = "sam")]
            return Err(lightcraft_segment::yolo::Model::load(stage).err().map_or_else(
                || "the YOLO segmentation model is invalid".into(),
                |e| e.to_string(),
            ));
            #[cfg(not(feature = "sam"))]
            Err("this build cannot validate YOLO segmentation models".into())
        }
        (true, true) => Err("choose a folder or archive containing only one model".into()),
        (false, false) => Err("the selected files are incomplete; SAM 3 needs model.safetensors, vocab.json and merges.txt, while YOLO needs model.onnx and lightcraft-yolo.json".into()),
    }
}

fn is_sam_layout(dir: &Path) -> bool {
    ["model.safetensors", "vocab.json", "merges.txt"].iter().all(|name| dir.join(name).is_file())
}

fn is_yolo_layout(dir: &Path) -> bool {
    dir.join(YOLO_MODEL).is_file() && dir.join(YOLO_MANIFEST).is_file()
}

fn canonical_file_name(name: &str) -> Option<&'static str> {
    MODEL_FILES.iter().copied().find(|candidate| name.eq_ignore_ascii_case(candidate))
}

fn file_limit(name: &str) -> u64 {
    match name {
        SAM_WEIGHTS => MAX_SAM_FILE_BYTES,
        YOLO_MODEL => MAX_YOLO_FILE_BYTES,
        _ => 16 << 20,
    }
}

fn source_name(paths: &[PathBuf]) -> String {
    let path = paths.first().map(PathBuf::as_path);
    let name = path
        .and_then(|p| if p.is_dir() { p.file_name() } else { p.file_stem() })
        .and_then(|n| n.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("ai-mask-model");
    name.to_string()
}

fn slug(name: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for ch in name.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            if out.len() < 48 {
                out.push(ch);
            }
            dash = false;
        } else if !dash && !out.is_empty() && out.len() < 48 {
            out.push('-');
            dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

pub fn row(model: &Model, selected: bool) -> Value {
    json!({
        "id": model.id,
        "kind": model.kind,
        "dir": model.dir.display().to_string(),
        "sizeBytes": model.size_bytes,
        "managed": model.managed,
        "selected": selected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    fn temp(tag: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("lc-ai-mask-model-store-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create test folder");
        path
    }

    fn sam_files(dir: &Path) {
        fs::write(dir.join(SAM_WEIGHTS), b"synthetic weights").expect("write weights");
        fs::write(dir.join("vocab.json"), b"{}").expect("write vocabulary");
        fs::write(dir.join("merges.txt"), b"merge").expect("write merges");
    }

    #[test]
    fn installs_sam_bundle_only_after_license_acceptance_and_persists_selection() {
        let base = temp("sam");
        let source = base.join("source");
        let store = base.join("managed");
        fs::create_dir(&source).expect("create source");
        sam_files(&source);
        let paths = ["model.safetensors", "vocab.json", "merges.txt"].map(|name| source.join(name).display().to_string());

        assert!(install(&store, &paths, false).is_err());
        assert_eq!(fs::read_dir(&store).expect("store exists").count(), 0);

        let installed = install(&store, &paths, true).expect("install model");
        assert_eq!(installed.id, "sam3");
        assert_eq!(installed.dir, store.join("sam3"));
        assert_eq!(installed.kind, "sam3");
        assert!(is_supported_model(&installed.dir));
        assert!(install(&store, &paths, true).is_err(), "SAM 3 always uses the sam3 folder");
        write_active(&store, Some(&installed.id)).expect("select model");
        assert_eq!(fs::read_to_string(store.join("active.txt")).expect("read active"), installed.id);
        assert_eq!(list(&store, None).expect("list models").len(), 1);
        assert!(remove(&store, &installed.id).is_ok());
        write_active(&store, None).expect("clear selection");
        assert!(!store.join("active.txt").exists());
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn installs_a_model_selected_as_a_folder() {
        let base = temp("folder");
        let source = base.join("my-sam-model");
        fs::create_dir(&source).expect("create source");
        sam_files(&source);
        let store = base.join("managed");
        let installed = install(&store, &[source.display().to_string()], true).expect("install folder");
        assert_eq!(installed.id, "sam3");
        assert_eq!(installed.dir, store.join("sam3"));
        assert_eq!(installed.kind, "sam3");
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn installs_sam_bundle_from_a_zip_and_rejects_traversal() {
        let base = temp("zip");
        let archive_path = base.join("sam.zip");
        let file = File::create(&archive_path).expect("create archive");
        let mut archive = zip::ZipWriter::new(file);
        for (name, bytes) in [
            ("checkpoint/model.safetensors", b"weights".as_slice()),
            ("checkpoint/vocab.json", b"{}".as_slice()),
            ("checkpoint/merges.txt", b"merge".as_slice()),
        ] {
            archive.start_file(name, SimpleFileOptions::default()).expect("start archive entry");
            archive.write_all(bytes).expect("write archive entry");
        }
        archive.finish().expect("finish archive");
        let store = base.join("managed");
        let installed = install(&store, &[archive_path.display().to_string()], true).expect("install ZIP model");
        assert!(is_supported_model(&installed.dir));

        let bad_path = base.join("bad.zip");
        let file = File::create(&bad_path).expect("create malicious archive");
        let mut archive = zip::ZipWriter::new(file);
        archive.start_file("../model.safetensors", SimpleFileOptions::default()).expect("start malicious entry");
        archive.write_all(b"weights").expect("write malicious entry");
        archive.finish().expect("finish malicious archive");
        assert!(extract_zip(&bad_path, &base.join("bad-stage")).is_err());
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn rejects_incomplete_bundles_and_invalid_model_ids() {
        let base = temp("invalid");
        let source = base.join("model.onnx");
        fs::write(&source, b"not a complete model").expect("write incomplete file");
        assert!(install(&base.join("managed"), &[source.display().to_string()], true).is_err());
        for id in ["", "../sam", "Upper", "-bad", "bad/thing"] {
            assert!(!valid_id(id), "unexpected valid id: {id}");
        }
        let _ = fs::remove_dir_all(base);
    }

    #[cfg(feature = "sam")]
    #[test]
    fn refuses_yolo_files_that_are_not_loadable_onnx_models() {
        let base = temp("invalid-yolo");
        fs::write(base.join(YOLO_MODEL), b"not an ONNX graph").expect("write invalid graph");
        fs::write(base.join(YOLO_MANIFEST), br#"{"format":"lightcraft-yolo-seg","version":1,"image_size":640,"labels":["person"]}"#)
            .expect("write manifest");
        assert!(validate_stage(&base).is_err());
        let _ = fs::remove_dir_all(base);
    }
}
