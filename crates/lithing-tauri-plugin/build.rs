const COMMANDS: &[&str] = &["start", "stop", "update_settings", "calibrate", "status", "list_cameras", "set_keep_frame_rate", "list_camera_controls", "capture_level", "list_camera_formats"];

fn main() {
    // Generates the allow-/deny- permissions for every command (Tauri 2 ACL).
    // android/: the Camera2 capture and the model copy (LithingCameraPlugin.kt), built into the app.
    tauri_plugin::Builder::new(COMMANDS).android_path("android").build();
    // Test executables link tauri, which imports comctl32 v6 functions: without the
    // Common-Controls manifest Windows refuses to start them (STATUS_ENTRYPOINT_NOT_FOUND).
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("windows-msvc") {
        println!("cargo:rustc-link-arg-tests=/MANIFEST:EMBED");
        println!(
            "cargo:rustc-link-arg-tests=/MANIFESTDEPENDENCY:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'"
        );
    }
}
