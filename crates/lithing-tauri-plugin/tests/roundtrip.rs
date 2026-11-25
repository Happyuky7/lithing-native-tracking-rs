//! Proves the round trip: tracker -> plugin -> Channel -> binary frame ->
//! guest-js decoder, and the plugin's commands through Tauri's IPC (mock runtime).
//!
//! The tracking parts need the ONNX models (scripts/fetch_models.*) and are
//! skipped without them. Set LITHING_TEST_VIDEO to a clip with a face for a
//! content check; otherwise a generated grey clip is used.

use lithing_tracking::models;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::ipc::{Channel, InvokeBody, InvokeResponseBody};
use tauri_plugin_lithing_tracking::{StartOptions, Status, TrackingState, start_tracking, stop_tracking, wire};

fn models_dir() -> Option<PathBuf> {
    let dir = models::default_model_dir();
    models::ALL_MODELS.iter().all(|m| dir.join(m).is_file()).then_some(dir)
}

/// A short grey 4:2:0 clip (no face): frames still flow through every stage.
fn grey_clip(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("lithing-grey-{name}-{}.y4m", std::process::id()));
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(b"YUV4MPEG2 W320 H240 F30:1 Ip C420jpeg\n").unwrap();
    for i in 0..60u32 {
        file.write_all(b"FRAME\n").unwrap();
        file.write_all(&vec![(60 + i) as u8; 320 * 240]).unwrap();
        file.write_all(&vec![128u8; 2 * 160 * 120]).unwrap();
    }
    path
}

fn test_video(name: &str) -> PathBuf {
    std::env::var_os("LITHING_TEST_VIDEO").map(PathBuf::from).unwrap_or_else(|| grey_clip(name))
}

#[test]
fn frames_reach_the_typescript_decoder() {
    let Some(dir) = models_dir() else {
        eprintln!("skipped: models missing (run scripts/fetch_models)");
        return;
    };
    let payloads: Arc<Mutex<Vec<Vec<u8>>>> = Arc::default();
    let sink = payloads.clone();
    let channel = Channel::new(move |body| {
        if let InvokeResponseBody::Raw(bytes) = body {
            sink.lock().unwrap().push(bytes);
        }
        Ok(())
    });
    let state = TrackingState::default();
    let mut options = StartOptions { video_file: Some(test_video("channel").display().to_string()), provider: Some("cpu".into()), models_dir: Some(dir.display().to_string()), ..Default::default() };
    options.settings.hands_enabled = true;
    let status = start_tracking(&state, options, None, channel).expect("start");
    assert!(status.width > 0);
    let deadline = Instant::now() + Duration::from_secs(30);
    while payloads.lock().unwrap().len() < 30 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    stop_tracking(&state);
    let payloads = payloads.lock().unwrap().clone();
    assert!(payloads.len() >= 30, "only {} frames arrived", payloads.len());
    let decoded: Vec<_> = payloads.iter().map(|p| wire::decode(p).expect("decode")).collect();
    assert!(decoded.windows(2).all(|w| w[0].frame_id < w[1].frame_id), "frames out of order");
    assert!(payloads.iter().all(|p| p.len() < 1024), "a frame exceeded Tauri's direct-delivery size");

    // The same bytes through the TypeScript decoder (Node strips the types).
    let record = std::env::temp_dir().join(format!("lithing-frames-{}.bin", std::process::id()));
    let mut file = std::fs::File::create(&record).unwrap();
    for payload in &payloads {
        file.write_all(&(payload.len() as u32).to_le_bytes()).unwrap();
        file.write_all(payload).unwrap();
    }
    drop(file);
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("guest-js/roundtrip-test.mjs");
    let output = match std::process::Command::new("node").arg(&script).arg(&record).output() {
        Ok(output) if output.status.success() => output,
        Ok(output) => panic!("node decoder failed: {}", String::from_utf8_lossy(&output.stderr)),
        Err(_) => {
            eprintln!("node not found: TypeScript half skipped");
            return;
        }
    };
    let js: Vec<serde_json::Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(js.len(), decoded.len());
    for (js, rust) in js.iter().zip(&decoded) {
        assert_eq!(js["frameId"].as_u64().unwrap(), rust.frame_id);
        assert_eq!(js["hands"].as_array().unwrap().len(), rust.hands.len());
        if rust.flags & wire::FLAG_TRACKING != 0 {
            assert_eq!(js["tracking"]["headYaw"].as_f64().unwrap() as f32, rust.tracking.head_yaw);
            assert_eq!(js["tracking"]["blinkLeft"].as_f64().unwrap() as f32, rust.tracking.blink_left);
        }
        if rust.flags & wire::FLAG_BLENDSHAPES != 0 {
            assert_eq!(js["blendshapes"]["jawOpen"].as_f64().unwrap() as f32, rust.blendshapes[25]);
        }
        if let Some(eyes) = &rust.eyes {
            assert_eq!(js["eyes"]["right"]["gap"].as_f64().unwrap() as f32, eyes.right.gap);
            assert_eq!(js["eyes"]["left"]["output"].as_f64().unwrap() as f32, eyes.left.output);
            assert_eq!(js["eyes"]["left"]["calibrated"].as_bool().unwrap(), eyes.left.calibrated);
        }
        if let Some(gaze) = &rust.gaze {
            assert_eq!(js["gaze"]["x"].as_f64().unwrap() as f32, gaze.x);
            assert_eq!(js["gaze"]["y"].as_f64().unwrap() as f32, gaze.y);
        }
        for (jh, rh) in js["hands"].as_array().unwrap().iter().zip(&rust.hands) {
            assert_eq!(jh["handedness"].as_str().unwrap(), rh.handedness.name());
            assert_eq!(jh["wristX"].as_f64().unwrap() as f32, rh.wrist_x);
            assert_eq!(jh["fingers"][7].as_f64().unwrap() as f32, rh.fingers[7]);
        }
    }
    if std::env::var_os("LITHING_TEST_VIDEO").is_some() {
        assert!(decoded.iter().any(|d| d.flags & wire::FLAG_BLENDSHAPES != 0), "no face found in LITHING_TEST_VIDEO");
        assert!(decoded.iter().any(|d| !d.hands.is_empty()), "no hands found in LITHING_TEST_VIDEO");
    }
    eprintln!("{} frames decoded identically in Rust and TypeScript", decoded.len());
}

fn request(cmd: &str, body: serde_json::Value) -> tauri::webview::InvokeRequest {
    tauri::webview::InvokeRequest {
        cmd: cmd.into(),
        callback: tauri::ipc::CallbackFn(0),
        error: tauri::ipc::CallbackFn(1),
        url: if cfg!(any(windows, target_os = "android")) { "http://tauri.localhost" } else { "tauri://localhost" }.parse().unwrap(),
        body: InvokeBody::Json(body),
        headers: Default::default(),
        invoke_key: tauri::test::INVOKE_KEY.to_string(),
    }
}

#[test]
fn commands_work_through_ipc() {
    // The mock context has no capability files: allow the plugin's commands the way
    // `lithing-tracking:default` does in an app's capabilities/default.json.
    let mut context = tauri::test::mock_context(tauri::test::noop_assets());
    for command in ["start", "stop", "update_settings", "calibrate", "status", "list_cameras"] {
        context.runtime_authority_mut().__allow_command(format!("plugin:lithing-tracking|{command}"), tauri::utils::acl::ExecutionContext::Local);
    }
    let app = tauri::test::mock_builder().plugin(tauri_plugin_lithing_tracking::init()).build(context).expect("app");
    let webview = tauri::WebviewWindowBuilder::new(&app, "main", Default::default()).build().unwrap();
    let status = |webview: &tauri::WebviewWindow<tauri::test::MockRuntime>| -> Status {
        tauri::test::get_ipc_response(webview, request("plugin:lithing-tracking|status", serde_json::json!({}))).unwrap().deserialize().unwrap()
    };
    assert!(!status(&webview).running);
    let Some(dir) = models_dir() else {
        eprintln!("skipped the start/stop half: models missing");
        return;
    };
    // The web view's Channel arrives as "__CHANNEL__:<id>"; in the mock runtime its messages go nowhere.
    let options = serde_json::json!({ "videoFile": test_video("ipc"), "provider": "cpu", "modelsDir": dir, "settings": { "handsEnabled": true } });
    let started = tauri::test::get_ipc_response(&webview, request("plugin:lithing-tracking|start", serde_json::json!({ "options": options, "onFrame": "__CHANNEL__:7" })));
    let started: Status = started.expect("start").deserialize().unwrap();
    assert!(started.width > 0);
    let deadline = Instant::now() + Duration::from_secs(30);
    while status(&webview).frames_sent < 10 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(status(&webview).frames_sent >= 10);
    tauri::test::get_ipc_response(&webview, request("plugin:lithing-tracking|update_settings", serde_json::json!({ "settings": { "faceFps": 15 } }))).expect("update_settings");
    // The eye openness settings as the app's client sends them.
    let cue = serde_json::json!({ "open": 0.28, "closed": 0.1 });
    let eye = serde_json::json!({ "blink": { "open": 0.1, "closed": 0.6 }, "gap": cue, "iris": { "open": 0.6, "closed": 0.2 } });
    let face = serde_json::json!({ "mirror": true, "glassesMode": true, "lowLight": false, "highLight": false, "stability": 0.72, "eyeOpenness": 0.7, "eyeCalibration": { "right": eye, "left": eye, "capturedAt": 1.0 } });
    tauri::test::get_ipc_response(&webview, request("plugin:lithing-tracking|update_settings", serde_json::json!({ "settings": { "faceFps": 15, "face": face } }))).expect("update_settings with eyes");
    // And the pupil calibration on top, as the app's client sends it.
    let mut face = face;
    face["gazeCalibration"] = serde_json::json!({ "center": { "x": 0.02, "y": -0.04 }, "left": -0.3, "right": 0.28, "up": 0.2, "down": -0.25, "capturedAt": 2.0 });
    tauri::test::get_ipc_response(&webview, request("plugin:lithing-tracking|update_settings", serde_json::json!({ "settings": { "faceFps": 15, "face": face } }))).expect("update_settings with the gaze");
    tauri::test::get_ipc_response(&webview, request("plugin:lithing-tracking|calibrate", serde_json::json!({}))).expect("calibrate");
    tauri::test::get_ipc_response(&webview, request("plugin:lithing-tracking|stop", serde_json::json!({}))).expect("stop");
    assert!(!status(&webview).running);
}
