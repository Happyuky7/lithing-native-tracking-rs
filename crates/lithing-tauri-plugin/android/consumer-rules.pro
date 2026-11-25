# The Rust side calls into these by name (JNI) and Tauri finds the plugin by reflection.
-keep class com.lithing.tracking.LithingNative { *; }
-keep class com.lithing.tracking.LithingCameraPlugin { *; }
-keep class com.lithing.tracking.StartCameraArgs { *; }
