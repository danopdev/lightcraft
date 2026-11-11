//! Settings ▸ AI Mask: installing, choosing and removing local SAM 3 / YOLO segmentation models.

use egui::RichText;
use serde_json::{Value, json};

use crate::LightcraftApp;
use crate::theme::Tokens;
use crate::widgets::register;

const REFRESH_SECS: f64 = 1.5;
const INSTALL_TASK: &str = "Install AI Mask Model";

#[derive(Default)]
pub struct Ui {
    epoch: u64,
    list: Option<(u64, f64, Value)>,
    pub acknowledged_sam: bool,
    error: Option<String>,
}

fn models(app: &mut LightcraftApp) -> Result<Value, String> {
    let now = app.tick_now();
    let epoch = app.caches.ai_masks.epoch;
    if let Some((cached_epoch, at, value)) = &app.caches.ai_masks.list
        && *cached_epoch == epoch
        && now - at < REFRESH_SECS
    {
        return Ok(value.clone());
    }
    let result = app.session.execute("segment.models.list", &json!({})).map_err(|e| e.to_string())?;
    app.caches.ai_masks.list = Some((epoch, now, result.clone()));
    Ok(result)
}

/// The model manager in Settings ▸ AI Mask.
pub fn settings_tab(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens) {
    ui.label(RichText::new(crate::i18n::tr("AI Mask")).font(t.semibold(15.0)).color(t.text));
    if !lightcraft_engine::segment::Segmenter::AVAILABLE {
        ui.label(RichText::new(crate::i18n::tr("AI Mask models are not supported in this build.")).color(t.text_dim));
        return;
    }
    let list = match models(app) {
        Ok(list) => list,
        Err(error) => {
            ui.label(RichText::new(error).color(t.caution));
            return;
        }
    };
    if let Some(dir) = list["dir"].as_str() {
        ui.label(RichText::new(crate::i18n::tr_format!("Installed models are kept in {dir}", dir = dir)).color(t.text_dim));
    } else {
        ui.label(RichText::new(crate::i18n::tr("This build has nowhere to keep AI Mask models.")).color(t.caution));
        return;
    }
    if list["lockedByEnvironment"] == true {
        ui.label(
            RichText::new(crate::i18n::tr(
                "The active model is fixed by LIGHTCRAFT_SEGMENTATION_MODEL_DIR or LIGHTCRAFT_SAM3_DIR; installed models cannot be selected.",
            ))
            .color(t.text_dim),
        );
    }
    ui.label(crate::i18n::tr(
        "Install SAM 3 files or a converted YOLO segmentation model. Large files and archives are checked and copied in the background.",
    ));

    ui.horizontal_wrapped(|ui| {
        let busy = app.tasks.is_running(INSTALL_TASK) || app.tasks.is_running("Remove AI Mask Model");
        let files = crate::button::Button::new("Install files or ZIP…")
            .enabled(!busy && app.services.pick_segmentation_model_files.is_some())
            .widget("aiMasks:installFiles")
            .show(ui);
        if files.clicked()
            && let Some(paths) = app.services.pick_segmentation_model_files.as_mut().map(|pick| pick())
            && !paths.is_empty()
        {
            start_install(app, paths);
        }
        let folder = crate::button::Button::new("Install a folder…")
            .enabled(!busy && app.services.pick_folder.is_some())
            .widget("aiMasks:installFolder")
            .show(ui);
        if folder.clicked()
            && let Some(path) = app.services.pick_folder.as_mut().and_then(|pick| pick())
        {
            start_install(app, vec![path]);
        }
    });

    if let Some(error) = app.caches.ai_masks.error.as_deref() {
        ui.label(RichText::new(error).color(t.caution));
    }
    let models = list["models"].as_array().cloned().unwrap_or_default();
    if models.is_empty() {
        ui.label(RichText::new(crate::i18n::tr("No AI Mask models are installed.")).color(t.text_dim));
    }
    for model in &models {
        model_row(app, ui, t, model, list["lockedByEnvironment"] == true);
    }
}

fn start_install(app: &mut LightcraftApp, paths: Vec<String>) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let Some(store) = app.session.segmenter.models_dir.clone() else {
            app.caches.ai_masks.error = Some(crate::i18n::tr("This build has nowhere to keep AI Mask models.").into());
            return;
        };
        let acknowledged = true;
        app.caches.ai_masks.error = None;
        if let Err(error) = crate::tasks::spawn(
            app,
            INSTALL_TASK,
            Some("model"),
            move || {
                let mut session = lightcraft_engine::Session::new();
                session.segmenter.models_dir = Some(store);
                session.execute("segment.models.install", &json!({"paths": paths, "acknowledgedSam": acknowledged})).map_err(|e| e.to_string())
            },
            |app, ctx, result| {
                app.caches.ai_masks.epoch += 1;
                app.caches.ai_masks.acknowledged_sam = true;
                match result {
                    Ok(model) => {
                        app.caches.ai_masks.error = None;
                        let id = model["id"].as_str().unwrap_or(crate::i18n::tr("model"));
                        app.toast(ctx, crate::i18n::tr_format!("Installed AI Mask model: {id}", id = id));
                    }
                    Err(error) => {
                        app.caches.ai_masks.error = Some(error.clone());
                        app.toast_error(ctx, error);
                    }
                }
            },
        ) {
            app.caches.ai_masks.error = Some(error);
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = paths;
        app.caches.ai_masks.error = Some(crate::i18n::tr("AI Mask model installation is available in the desktop app only.").into());
    }
}

fn model_row(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, model: &Value, locked: bool) {
    let id = model["id"].as_str().unwrap_or_default().to_string();
    let selected = model["selected"] == true;
    let managed = model["managed"] == true;
    let busy = app.tasks.is_running(INSTALL_TASK) || app.tasks.is_running("Remove AI Mask Model");
    let kind = match model["kind"].as_str() {
        Some("sam3") => "SAM 3",
        Some("yolo") => "YOLO segmentation",
        _ => "Segmentation",
    };
    ui.horizontal_wrapped(|ui| {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(&id).color(t.text));
                ui.label(RichText::new(crate::i18n::tr(kind)).color(t.text_dim));
                if let Some(size) = model["sizeBytes"].as_u64() {
                    ui.label(RichText::new(format_size(size)).color(t.text_dim));
                }
            });
            if let Some(dir) = model["dir"].as_str() {
                ui.label(RichText::new(dir).font(t.font(11.0)).color(t.text_dim));
            }
        });
        if !selected {
            let activate = ui.add_enabled(!locked && managed && !busy, egui::Button::new(crate::i18n::tr("Active")));
            register(ui, format!("aiMasks:active:{id}"), activate.rect);
            if activate.clicked() {
                run_model_command(app, "segment.models.select", json!({"id": id}));
            }
        }
        let remove = ui.add_enabled(managed && !(locked && selected) && !busy, egui::Button::new(crate::i18n::tr("Remove")));
        register(ui, format!("aiMasks:remove:{id}"), remove.rect);
        if remove.clicked() {
            start_remove(app, id);
        }
    });
    ui.separator();
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 { format!("{:.1} GB", bytes as f64 / 1_000_000_000.0) } else { format!("{:.0} MB", bytes as f64 / 1_000_000.0) }
}

fn run_model_command(app: &mut LightcraftApp, command: &str, params: Value) {
    match app.run(command, params) {
        Ok(_) => {
            app.caches.ai_masks.epoch += 1;
            app.caches.ai_masks.error = None;
        }
        Err(error) => app.caches.ai_masks.error = Some(error),
    }
}

fn start_remove(app: &mut LightcraftApp, id: String) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let Some(store) = app.session.segmenter.models_dir.clone() else {
            app.caches.ai_masks.error = Some(crate::i18n::tr("This build has nowhere to keep AI Mask models.").into());
            return;
        };
        if app.session.segmenter.dir.as_deref() == Some(store.join(&id).as_path()) {
            if lightcraft_engine::config::segmentation_model_override_set() {
                app.caches.ai_masks.error =
                    Some(crate::i18n::tr("The active model is fixed by an environment variable and cannot be removed.").into());
                return;
            }
            if let Err(error) = app.run("segment.models.select", json!({"id": null})) {
                app.caches.ai_masks.error = Some(error);
                return;
            }
            app.caches.ai_masks.epoch += 1;
            app.caches.ai_masks.error = None;
            if app.session.segmenter.dir.as_deref() == Some(store.join(&id).as_path()) {
                app.session.segmenter.dir = None;
            }
        }
        let active = app.session.segmenter.dir.clone();
        if let Err(error) = crate::tasks::spawn(
            app,
            "Remove AI Mask Model",
            Some("model"),
            move || {
                let mut session = lightcraft_engine::Session::new();
                session.segmenter.models_dir = Some(store);
                session.segmenter.dir = active;
                session.execute("segment.models.remove", &json!({"id": id})).map(|_| ()).map_err(|e| e.to_string())
            },
            |app, ctx, result| {
                app.caches.ai_masks.epoch += 1;
                match result {
                    Ok(()) => app.caches.ai_masks.error = None,
                    Err(error) => {
                        app.caches.ai_masks.error = Some(error.clone());
                        app.toast_error(ctx, error);
                    }
                }
            },
        ) {
            app.caches.ai_masks.error = Some(error);
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = id;
        app.caches.ai_masks.error = Some(crate::i18n::tr("AI Mask model removal is available in the desktop app only.").into());
    }
}
