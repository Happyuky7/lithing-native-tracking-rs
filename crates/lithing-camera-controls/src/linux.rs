//! Linux backend: V4L2 controls on `/dev/videoN` (the node nokhwa opened as
//! index N). Opening a V4L2 node to read or write controls does not start or
//! disturb a stream, whoever owns it.
//!
//! * `V4L2_CID_EXPOSURE_AUTO_PRIORITY` (`exposure_auto_priority`, called
//!   `exposure_dynamic_framerate` by newer kernels): 0 keeps the frame rate.
//! * `V4L2_CID_EXPOSURE_AUTO` (`auto_exposure`) + `V4L2_CID_EXPOSURE_ABSOLUTE`
//!   (`exposure_time_absolute`, 100 us), presented as one [`ControlKind::Exposure`]
//!   with an automatic flag; the automatic mode the camera had is remembered.
//! * `V4L2_CID_GAIN`, `V4L2_CID_BRIGHTNESS`, `V4L2_CID_BACKLIGHT_COMPENSATION`.

use crate::{CameraControls, ControlInfo, ControlKind, ControlValue, DeviceEntry, ExposureUnits};
use std::ffi::CString;

const V4L2_CID_BRIGHTNESS: u32 = 0x0098_0900;
const V4L2_CID_GAIN: u32 = 0x0098_0913;
const V4L2_CID_BACKLIGHT_COMPENSATION: u32 = 0x0098_091c;
const V4L2_CID_EXPOSURE_AUTO: u32 = 0x009a_0901;
const V4L2_CID_EXPOSURE_ABSOLUTE: u32 = 0x009a_0902;
const V4L2_CID_EXPOSURE_AUTO_PRIORITY: u32 = 0x009a_0903;
const V4L2_CTRL_FLAG_DISABLED: u32 = 0x0001;
const V4L2_CTRL_FLAG_READ_ONLY: u32 = 0x0004;
/// `V4L2_EXPOSURE_AUTO`, `_MANUAL`, `_SHUTTER_PRIORITY`, `_APERTURE_PRIORITY`.
const EXPOSURE_MANUAL: i32 = 1;
const EXPOSURE_APERTURE_PRIORITY: i32 = 3;
const EXPOSURE_AUTO: i32 = 0;

#[repr(C)]
#[derive(Default)]
struct V4l2Queryctrl {
    id: u32,
    kind: u32,
    name: [u8; 32],
    minimum: i32,
    maximum: i32,
    step: i32,
    default_value: i32,
    flags: u32,
    reserved: [u32; 2],
}

#[repr(C)]
#[derive(Default)]
struct V4l2Control {
    id: u32,
    value: i32,
}

#[repr(C, packed)]
struct V4l2Querymenu {
    id: u32,
    index: u32,
    name: [u8; 32],
    reserved: u32,
}

const fn iowr(nr: u32, size: usize) -> u64 {
    // _IOWR('V', nr, size): dir 3 (read | write) << 30, size << 16, type << 8, nr.
    (3u64 << 30) | ((size as u64) << 16) | ((b'V' as u64) << 8) | nr as u64
}

const VIDIOC_G_CTRL: u64 = iowr(27, std::mem::size_of::<V4l2Control>());
const VIDIOC_S_CTRL: u64 = iowr(28, std::mem::size_of::<V4l2Control>());
const VIDIOC_QUERYCTRL: u64 = iowr(36, std::mem::size_of::<V4l2Queryctrl>());
const VIDIOC_QUERYMENU: u64 = iowr(37, std::mem::size_of::<V4l2Querymenu>());

/// Capture nodes (`/dev/videoN` with a name in sysfs); nothing is opened.
pub fn list_devices() -> Result<Vec<DeviceEntry>, String> {
    let mut list = Vec::new();
    let entries = std::fs::read_dir("/sys/class/video4linux").map_err(|e| format!("/sys/class/video4linux: {e}"))?;
    for entry in entries.flatten() {
        let file = entry.file_name().to_string_lossy().into_owned();
        let Some(index) = file.strip_prefix("video").and_then(|n| n.parse::<u32>().ok()) else { continue };
        let name = std::fs::read_to_string(entry.path().join("name")).unwrap_or_default().trim().to_string();
        let driver = std::fs::read_link(entry.path().join("device/driver")).ok().and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())).unwrap_or_default();
        let id = format!("/dev/video{index}");
        let virtual_camera = crate::is_virtual(&name, &driver) || driver.contains("v4l2loopback");
        list.push(DeviceEntry { index, name, id, virtual_camera });
    }
    list.sort_by_key(|d| d.index);
    Ok(list)
}

pub struct V4l2Controls {
    fd: i32,
    path: String,
    /// The automatic exposure mode to go back to (the camera's, when it was automatic at open).
    auto_mode: Option<i32>,
}

impl V4l2Controls {
    pub fn open(index: u32) -> Result<Self, String> {
        let path = format!("/dev/video{index}");
        let c_path = CString::new(path.clone()).map_err(|e| e.to_string())?;
        let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDWR | libc::O_NONBLOCK | libc::O_CLOEXEC) };
        if fd < 0 {
            return Err(format!("{path}: {}", std::io::Error::last_os_error()));
        }
        let mut controls = Self { fd, path, auto_mode: None };
        if let Some((_, mode)) = controls.read(V4L2_CID_EXPOSURE_AUTO).ok().flatten()
            && mode != EXPOSURE_MANUAL {
                controls.auto_mode = Some(mode);
            }
        Ok(controls)
    }

    fn ioctl<T>(&self, request: u64, arg: &mut T) -> std::io::Result<()> {
        let result = unsafe { libc::ioctl(self.fd, request as _, arg as *mut T) };
        if result < 0 { Err(std::io::Error::last_os_error()) } else { Ok(()) }
    }

    /// The control's description and value; None if the driver has no such control.
    fn read(&self, id: u32) -> std::io::Result<Option<(V4l2Queryctrl, i32)>> {
        let mut query = V4l2Queryctrl { id, ..Default::default() };
        match self.ioctl(VIDIOC_QUERYCTRL, &mut query) {
            Ok(()) => {}
            Err(e) if e.raw_os_error() == Some(libc::EINVAL) => return Ok(None),
            Err(e) => return Err(e),
        }
        if query.flags & V4L2_CTRL_FLAG_DISABLED != 0 {
            return Ok(None);
        }
        let mut control = V4l2Control { id, value: 0 };
        self.ioctl(VIDIOC_G_CTRL, &mut control)?;
        Ok(Some((query, control.value)))
    }

    fn write(&self, id: u32, value: i32) -> std::io::Result<()> {
        let mut control = V4l2Control { id, value };
        self.ioctl(VIDIOC_S_CTRL, &mut control)
    }

    fn menu_has(&self, id: u32, index: i32) -> bool {
        let mut menu = V4l2Querymenu { id, index: index as u32, name: [0; 32], reserved: 0 };
        self.ioctl(VIDIOC_QUERYMENU, &mut menu).is_ok()
    }

    /// The automatic exposure mode to use: the camera's own, else aperture priority (UVC), else full auto.
    fn auto_mode(&self) -> i32 {
        self.auto_mode.unwrap_or_else(|| if self.menu_has(V4L2_CID_EXPOSURE_AUTO, EXPOSURE_APERTURE_PRIORITY) { EXPOSURE_APERTURE_PRIORITY } else { EXPOSURE_AUTO })
    }

    fn name(query: &V4l2Queryctrl) -> String {
        let end = query.name.iter().position(|&b| b == 0).unwrap_or(query.name.len());
        String::from_utf8_lossy(&query.name[..end]).into_owned()
    }
}

impl Drop for V4l2Controls {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd) };
    }
}

fn plain(kind: ControlKind) -> Option<u32> {
    match kind {
        ControlKind::AutoExposurePriority => Some(V4L2_CID_EXPOSURE_AUTO_PRIORITY),
        ControlKind::Gain => Some(V4L2_CID_GAIN),
        ControlKind::Brightness => Some(V4L2_CID_BRIGHTNESS),
        ControlKind::BacklightCompensation => Some(V4L2_CID_BACKLIGHT_COMPENSATION),
        ControlKind::Exposure | ControlKind::VariableFrameRate => None,
    }
}

impl CameraControls for V4l2Controls {
    fn query(&mut self, kind: ControlKind) -> Result<Option<ControlInfo>, String> {
        let error = |e: std::io::Error| format!("{}: {e}", self.path);
        if kind == ControlKind::Exposure {
            let Some((query, value)) = self.read(V4L2_CID_EXPOSURE_ABSOLUTE).map_err(error)? else { return Ok(None) };
            let mode = self.read(V4L2_CID_EXPOSURE_AUTO).map_err(error)?;
            let auto = mode.as_ref().is_some_and(|(_, m)| *m != EXPOSURE_MANUAL);
            return Ok(Some(ControlInfo {
                kind,
                name: format!("{} (+ auto_exposure)", Self::name(&query)),
                min: query.minimum as i64,
                max: query.maximum as i64,
                step: query.step as i64,
                default: query.default_value as i64,
                value: value as i64,
                auto,
                supports_auto: mode.is_some(),
                supports_manual: query.flags & V4L2_CTRL_FLAG_READ_ONLY == 0 && (mode.is_none() || self.menu_has(V4L2_CID_EXPOSURE_AUTO, EXPOSURE_MANUAL)),
            }));
        }
        let Some(id) = plain(kind) else { return Ok(None) };
        let Some((query, value)) = self.read(id).map_err(error)? else { return Ok(None) };
        Ok(Some(ControlInfo {
            kind,
            name: Self::name(&query),
            min: query.minimum as i64,
            max: query.maximum as i64,
            step: query.step as i64,
            default: query.default_value as i64,
            value: value as i64,
            auto: false,
            supports_auto: false,
            supports_manual: query.flags & V4L2_CTRL_FLAG_READ_ONLY == 0,
        }))
    }

    fn set(&mut self, kind: ControlKind, value: ControlValue) -> Result<(), String> {
        let v = value.value.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        let error = |e: std::io::Error| format!("{}: {e}", self.path);
        if kind == ControlKind::Exposure {
            if value.auto {
                // Value first (while still manual, where drivers accept it), then the automatic mode.
                let _ = self.write(V4L2_CID_EXPOSURE_ABSOLUTE, v);
                return self.write(V4L2_CID_EXPOSURE_AUTO, self.auto_mode()).map_err(error);
            }
            if let Some((_, mode)) = self.read(V4L2_CID_EXPOSURE_AUTO).map_err(error)?
                && mode != EXPOSURE_MANUAL {
                    self.auto_mode.get_or_insert(mode);
                    self.write(V4L2_CID_EXPOSURE_AUTO, EXPOSURE_MANUAL).map_err(error)?;
                }
            return self.write(V4L2_CID_EXPOSURE_ABSOLUTE, v).map_err(error);
        }
        let id = plain(kind).ok_or("not a V4L2 control")?;
        self.write(id, v).map_err(error)
    }

    fn exposure_units(&self) -> ExposureUnits {
        ExposureUnits::HundredMicroseconds
    }

    fn device_id(&self) -> String {
        self.path.clone()
    }
}
