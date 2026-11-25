//! Backward compatibility of the start options and status JSON.

use tauri_plugin_lithing_tracking::{StartOptions, Status, TrackingState, set_keep_frame_rate};

#[test]
fn keep_frame_rate_defaults_on_for_older_clients() {
    // What the app's copy of the client (before setKeepFrameRate) sends.
    let old: StartOptions = serde_json::from_str(r#"{ "camera": { "index": 1, "width": 640, "height": 480, "fps": 60 }, "provider": "auto" }"#).unwrap();
    assert!(old.camera.keep_frame_rate);
    assert_eq!(old.camera.index, 1);
    let off: StartOptions = serde_json::from_str(r#"{ "camera": { "index": 0, "width": 640, "height": 480, "fps": 60, "keepFrameRate": false } }"#).unwrap();
    assert!(!off.camera.keep_frame_rate);
    assert!(StartOptions::default().camera.keep_frame_rate);
}

#[test]
fn status_reports_camera_controls_in_camel_case() {
    let status = serde_json::to_value(Status::default()).unwrap();
    assert!(status.get("cameraControls").is_some_and(|v| v.is_null()));
    let report = lithing_tracking::camera_controls::CameraControlReport { strategy: "priority".into(), fps_before: Some(15.0), fps_after: Some(30.0), ..Default::default() };
    let status = Status { camera_controls: Some(report), ..Default::default() };
    let json = serde_json::to_value(&status).unwrap();
    assert_eq!(json["cameraControls"]["strategy"], "priority");
    assert_eq!(json["cameraControls"]["fpsBefore"], 15.0);
    assert_eq!(json["cameraControls"]["keepFrameRate"], false);
}

#[test]
fn live_toggle_without_a_camera_is_a_no_op() {
    assert!(!set_keep_frame_rate(&TrackingState::default(), false));
}

#[test]
fn camera_mode_is_auto_unless_given() {
    let old: StartOptions = serde_json::from_str(r#"{ "camera": { "index": 0, "width": 640, "height": 480, "fps": 60 } }"#).unwrap();
    assert!(old.camera_mode.is_none() && old.camera.mode.is_none());
    let auto: StartOptions = serde_json::from_str(r#"{ "cameraMode": null }"#).unwrap();
    assert!(auto.camera_mode.is_none());
    let mjpeg: StartOptions = serde_json::from_str(r#"{ "cameraMode": { "format": "MJPEG", "width": 1280, "height": 720, "fps": 30 } }"#).unwrap();
    assert_eq!(mjpeg.camera_mode.unwrap().to_string(), "MJPEG:1280x720@30");
    let any: StartOptions = serde_json::from_str(r#"{ "cameraMode": { "width": 640, "height": 480, "fps": 60 } }"#).unwrap();
    assert_eq!(any.camera_mode.unwrap().format, None);
}

#[test]
fn status_reports_format_mode_and_warning() {
    let json = serde_json::to_value(Status::default()).unwrap();
    assert_eq!(json["format"], "");
    assert!(json["cameraMode"].is_null() && json["cameraWarning"].is_null());
    // Status JSON of a plugin from before camera modes still reads.
    let old: Status = serde_json::from_str(
        r#"{ "running": true, "framesSent": 1, "width": 640, "height": 480, "fps": 30, "source": "cam", "faceProvider": "cpu", "handProvider": "cpu", "loadMs": 1, "dropped": 0, "lastError": null, "cameraControls": null }"#,
    )
    .unwrap();
    assert!(old.camera_mode.is_none() && old.format.is_empty());
}
