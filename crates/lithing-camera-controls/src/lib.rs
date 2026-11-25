//! Keeps a webcam at its frame rate in low light.
//!
//! Many UVC webcams lengthen their exposure in dim light ("auto exposure
//! priority", "low light compensation", "dynamic frame rate") and drop from
//! 30/60 to 15 fps or less, which a face tracker feels as lag. This crate
//! reads and writes the controls that cause it, next to the stream the
//! capture library opened, and puts every value back afterwards:
//!
//! * Windows (Media Foundation): `PROPSETID_VIDCAP_CAMERACONTROL` /
//!   `KSPROPERTY_CAMERACONTROL_AUTO_EXPOSURE_PRIORITY` through `IKsControl`,
//!   the extended `KSPROPERTY_CAMERACONTROL_EXTENDED_VFR` (variable frame
//!   rate) through `IMFExtendedCameraController`, and `IAMCameraControl`
//!   exposure / `IAMVideoProcAmp` gain, brightness and backlight compensation
//!   for the fallback ([`windows`]).
//! * Linux (V4L2): `exposure_auto_priority` (`exposure_dynamic_framerate`),
//!   `auto_exposure` + `exposure_time_absolute`, `gain`, `brightness` ([`linux`]).
//!
//! The policy is [`keep::KeepFrameRate`], a small state machine that only
//! talks to the [`CameraControls`] trait (unit-tested with a mock camera).

pub mod journal;
pub mod keep;
#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(windows)]
pub mod windows;

pub use keep::{KeepFrameRate, KeepFrameRateConfig};

/// The controls the policy knows about. Each backend maps them to its API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub enum ControlKind {
    /// UVC "auto-exposure priority": 1 lets auto exposure lower the frame rate. 0 = keep the rate.
    AutoExposurePriority,
    /// Windows extended control "variable frame rate" (value = KSCAMERA_EXTENDEDPROP_VFR_* flags).
    VariableFrameRate,
    /// Exposure time; `auto` = automatic exposure. Units: see [`ExposureUnits`].
    Exposure,
    Gain,
    Brightness,
    /// Reported only: it changes the metering target, not the frame rate.
    BacklightCompensation,
}

impl ControlKind {
    pub const ALL: [ControlKind; 6] = [Self::AutoExposurePriority, Self::VariableFrameRate, Self::Exposure, Self::Gain, Self::Brightness, Self::BacklightCompensation];

    pub fn key(self) -> &'static str {
        match self {
            Self::AutoExposurePriority => "auto-exposure-priority",
            Self::VariableFrameRate => "variable-frame-rate",
            Self::Exposure => "exposure",
            Self::Gain => "gain",
            Self::Brightness => "brightness",
            Self::BacklightCompensation => "backlight-compensation",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.key() == key)
    }
}

/// How a backend counts exposure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub enum ExposureUnits {
    /// DirectShow / KS: log2 of seconds (-5 = 1/32 s).
    Log2Seconds,
    /// V4L2 `exposure_time_absolute`: 100 microseconds.
    HundredMicroseconds,
}

impl ExposureUnits {
    pub fn seconds(self, value: i64) -> f64 {
        match self {
            Self::Log2Seconds => 2f64.powi(value.clamp(-30, 30) as i32),
            Self::HundredMicroseconds => value as f64 / 10_000.0,
        }
    }

    /// The longest exposure value of `info`'s range that still fits in `period` seconds.
    pub fn longest_within(self, period: f64, info: &ControlInfo) -> i64 {
        let wanted = match self {
            Self::Log2Seconds => period.log2().floor() as i64,
            Self::HundredMicroseconds => (period * 10_000.0).floor() as i64,
        };
        info.align_down(wanted)
    }
}

/// A control's current state as one value plus its automatic flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct ControlValue {
    pub value: i64,
    pub auto: bool,
}

impl ControlValue {
    pub fn manual(value: i64) -> Self {
        Self { value, auto: false }
    }
}

/// One control of a camera as the driver describes it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct ControlInfo {
    pub kind: ControlKind,
    /// The driver's name for it (V4L2) or the API's (Windows).
    pub name: String,
    pub min: i64,
    pub max: i64,
    pub step: i64,
    pub default: i64,
    pub value: i64,
    /// Currently automatic.
    pub auto: bool,
    /// Can be switched to automatic.
    pub supports_auto: bool,
    /// Can be set manually (some drivers expose read-only or auto-only controls).
    pub supports_manual: bool,
}

impl ControlInfo {
    pub fn current(&self) -> ControlValue {
        ControlValue { value: self.value, auto: self.auto }
    }

    /// Clamps into the range and onto the step grid (rounding down).
    pub fn align_down(&self, value: i64) -> i64 {
        let value = value.clamp(self.min, self.max.max(self.min));
        let step = self.step.max(1);
        let aligned = self.min + (value - self.min).div_euclid(step) * step;
        aligned.clamp(self.min, self.max.max(self.min))
    }
}

/// A camera's controls. Implemented by the Windows and Linux backends and by test mocks.
pub trait CameraControls {
    /// `Ok(None)`: the camera has no such control.
    fn query(&mut self, kind: ControlKind) -> Result<Option<ControlInfo>, String>;
    fn set(&mut self, kind: ControlKind, value: ControlValue) -> Result<(), String>;
    fn exposure_units(&self) -> ExposureUnits;
    /// Stable identity of the device (symbolic link / device node), for the restore journal.
    fn device_id(&self) -> String;
}

/// One write the policy made (or tried).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct ControlChange {
    pub kind: ControlKind,
    pub from: Option<ControlValue>,
    pub to: ControlValue,
    pub reason: String,
    pub error: Option<String>,
    /// Seconds since the stream opened.
    pub at: f64,
}

/// What a camera has, what was done to it and the frame rate it delivered.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, rename_all = "camelCase"))]
pub struct CameraControlReport {
    pub device: String,
    /// The option as currently set.
    pub keep_frame_rate: bool,
    /// A virtual camera (NVIDIA Broadcast, OBS, AMD Privacy View, ...): no hardware controls, nothing touched.
    pub virtual_camera: bool,
    pub controls: Vec<ControlInfo>,
    /// Why the controls could not be read (device busy in another app, no permission, ...).
    pub controls_error: Option<String>,
    /// "none", "priority" (auto exposure may no longer lower the rate),
    /// "exposure-cap" (manual exposure that fits the frame period, gain raised),
    /// "unsupported" (no control that helps), "off".
    pub strategy: String,
    pub changes: Vec<ControlChange>,
    /// Every changed control was put back (tracking stopped or the option turned off).
    pub restored: bool,
    pub restore_errors: Vec<String>,
    /// The negotiated rate.
    pub target_fps: f32,
    /// Delivered rate in the first seconds, before any control was changed.
    pub fps_before: Option<f32>,
    /// Delivered rate in the last second (after the changes).
    pub fps_after: Option<f32>,
    /// Mean luma (0-255) of the last frames, when measured.
    pub luma: Option<f32>,
    pub notes: Vec<String>,
}

/// A capture device as the OS lists it (nothing opened).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct DeviceEntry {
    /// The index the capture library (nokhwa) uses.
    pub index: u32,
    pub name: String,
    /// Symbolic link (Windows) or device node (Linux).
    pub id: String,
    pub virtual_camera: bool,
}

/// The controls of one listed device, read without starting a stream.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct DeviceControls {
    pub device: DeviceEntry,
    pub controls: Vec<ControlInfo>,
    /// Set when the device was not opened (virtual, busy) or could not be.
    pub skipped: Option<String>,
}

/// Name or path patterns of virtual cameras, which have no hardware controls.
const VIRTUAL_NAMES: &[&str] = &[
    "virtual", "obs", "nvidia broadcast", "privacy view", "snap camera", "xsplit", "droidcam", "iriun", "epoccam", "manycam", "camo", "streamlabs", "mmhmm", "vcam", "loopback", "elgato camera hub", "nvidia rtx",
];

/// Whether a device is a software camera (name hints, or a Windows symbolic
/// link that is not a bus device: `\\?\root#...`, `\\?\swd#...`).
pub fn is_virtual(name: &str, id: &str) -> bool {
    let name = name.to_ascii_lowercase();
    let id = id.to_ascii_lowercase();
    VIRTUAL_NAMES.iter().any(|hint| name.contains(hint)) || id.starts_with(r"\\?\root#") || id.starts_with(r"\\?\swd#") || id.contains("v4l2loopback")
}

/// Reads every known control of `controls` (missing ones are left out).
pub fn query_all(controls: &mut dyn CameraControls) -> (Vec<ControlInfo>, Vec<String>) {
    let mut found = Vec::new();
    let mut errors = Vec::new();
    for kind in ControlKind::ALL {
        match controls.query(kind) {
            Ok(Some(info)) => found.push(info),
            Ok(None) => {}
            Err(e) => errors.push(format!("{}: {e}", kind.key())),
        }
    }
    (found, errors)
}

/// Opens the controls of the camera the capture library opened as `index`
/// (Windows: matched by symbolic link `id` when given, else by index).
pub fn open(index: u32, id: &str) -> Result<Box<dyn CameraControls>, String> {
    #[cfg(windows)]
    return windows::MfControls::open(index, id).map(|c| Box::new(c) as Box<dyn CameraControls>);
    #[cfg(target_os = "linux")]
    {
        let _ = id;
        linux::V4l2Controls::open(index).map(|c| Box::new(c) as Box<dyn CameraControls>)
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        let _ = (index, id);
        Err("camera controls are not supported on this platform".into())
    }
}

/// Lists capture devices without opening any of them.
pub fn list_devices() -> Result<Vec<DeviceEntry>, String> {
    #[cfg(windows)]
    return windows::list_devices();
    #[cfg(target_os = "linux")]
    return linux::list_devices();
    #[cfg(not(any(windows, target_os = "linux")))]
    Ok(Vec::new())
}

/// Another application is using a camera right now (Windows: the privacy
/// settings' "currently in use" record). `None` when nothing is, or the OS
/// does not say (Linux: reading controls never interrupts a V4L2 stream).
pub fn camera_in_use() -> Option<String> {
    #[cfg(windows)]
    return windows::camera_in_use();
    #[cfg(not(windows))]
    None
}

/// Lists every device with its controls, without streaming. Virtual cameras
/// are not opened; physical ones are skipped while another app uses a camera
/// unless `force` (opening a busy device could interrupt it on some drivers).
pub fn list_device_controls(force: bool) -> Result<Vec<DeviceControls>, String> {
    let devices = list_devices()?;
    let busy = if force { None } else { camera_in_use() };
    Ok(devices
        .into_iter()
        .map(|device| {
            if device.virtual_camera {
                return DeviceControls { device, controls: Vec::new(), skipped: Some("virtual camera: no hardware controls, not opened".into()) };
            }
            if let Some(app) = &busy {
                return DeviceControls { device, controls: Vec::new(), skipped: Some(format!("not opened: a camera is in use by {app} (close it, or pass --force)")) };
            }
            match open(device.index, &device.id) {
                Ok(mut controls) => {
                    let (found, errors) = query_all(controls.as_mut());
                    let skipped = (!errors.is_empty()).then(|| errors.join("; "));
                    DeviceControls { device, controls: found, skipped }
                }
                Err(e) => DeviceControls { device, controls: Vec::new(), skipped: Some(format!("could not open: {e}")) },
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exposure(min: i64, max: i64, step: i64) -> ControlInfo {
        ControlInfo { kind: ControlKind::Exposure, name: "Exposure".into(), min, max, step, default: min, value: max, auto: true, supports_auto: true, supports_manual: true }
    }

    #[test]
    fn exposure_that_fits_the_frame_period() {
        // DirectShow units: 30 fps = 33.3 ms -> 2^-5 = 31.25 ms; 60 fps -> 2^-6.
        let log2 = exposure(-13, -1, 1);
        assert_eq!(ExposureUnits::Log2Seconds.longest_within(1.0 / 30.0, &log2), -5);
        assert_eq!(ExposureUnits::Log2Seconds.longest_within(1.0 / 60.0, &log2), -6);
        assert!(ExposureUnits::Log2Seconds.seconds(-5) <= 1.0 / 30.0);
        // V4L2 units (100 us), stepped: 333 -> 332 on a grid of 7 from 3.
        let v4l2 = exposure(3, 2047, 1);
        assert_eq!(ExposureUnits::HundredMicroseconds.longest_within(1.0 / 30.0, &v4l2), 333);
        let stepped = ControlInfo { step: 7, ..v4l2.clone() };
        assert_eq!(ExposureUnits::HundredMicroseconds.longest_within(1.0 / 30.0, &stepped), 332);
        assert!(ExposureUnits::HundredMicroseconds.seconds(ExposureUnits::HundredMicroseconds.longest_within(1.0 / 30.0, &stepped)) <= 1.0 / 30.0);
        // A range that cannot go that short: the shortest it has.
        assert_eq!(ExposureUnits::Log2Seconds.longest_within(1.0 / 60.0, &exposure(-4, 0, 1)), -4);
    }

    #[test]
    fn virtual_cameras_are_recognised() {
        assert!(is_virtual("NVIDIA Broadcast", r"\\?\root#image#0000#{e5323777}"));
        assert!(is_virtual("OBS Virtual Camera", r"\\?\swd#vcamdevapi#{guid}"));
        assert!(is_virtual("AMD Privacy View", r"\\?\something"));
        assert!(!is_virtual("USB Camera", r"\\?\usb#vid_0c45&pid_636b&mi_00#7&1a2b&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\global"));
        assert!(!is_virtual("HD Pro Webcam C920", "/dev/video0"));
    }

    #[test]
    fn kind_keys_round_trip() {
        for kind in ControlKind::ALL {
            assert_eq!(ControlKind::from_key(kind.key()), Some(kind));
        }
    }
}
