//! Windows backend: the camera's own Media Foundation source, opened next to
//! the capture library's stream (nokhwa keeps its source private and its
//! `CameraControl` API cannot switch exposure between automatic and manual
//! nor reach the auto-exposure priority, so the controls are read and written
//! on a second source of the same device, matched by symbolic link). Control
//! writes go to the driver, so they act on the running stream.
//!
//! * `IKsControl`, `PROPSETID_VIDCAP_CAMERACONTROL`:
//!   `KSPROPERTY_CAMERACONTROL_AUTO_EXPOSURE_PRIORITY` (UVC CT_AE_PRIORITY_CONTROL).
//! * `IMFExtendedCameraController`: `KSPROPERTY_CAMERACONTROL_EXTENDED_VFR`
//!   (drivers / device MFTs that implement extended controls; plain UVC
//!   cameras usually do not).
//! * `IAMCameraControl`: `CameraControl_Exposure` (log2 seconds, auto / manual flag).
//! * `IAMVideoProcAmp`: `VideoProcAmp_Gain`, `_Brightness`, `_BacklightCompensation`.

use crate::{CameraControls, ControlInfo, ControlKind, ControlValue, DeviceEntry, ExposureUnits};
use std::ffi::c_void;
use windows::Win32::Foundation::ERROR_SUCCESS;
use windows::Win32::Media::DirectShow::{
    CameraControl_Exposure, CameraControl_Flags_Auto, CameraControl_Flags_Manual, IAMCameraControl, IAMVideoProcAmp, VideoProcAmp_BacklightCompensation, VideoProcAmp_Brightness, VideoProcAmp_Flags_Auto, VideoProcAmp_Flags_Manual, VideoProcAmp_Gain,
};
use windows::Win32::Media::KernelStreaming::{
    IKsControl, KSIDENTIFIER_0_0, KSPROPERTY_CAMERACONTROL_AUTO_EXPOSURE_PRIORITY, KSPROPERTY_CAMERACONTROL_EXTENDED_VFR, KSPROPERTY_CAMERACONTROL_FLAGS_MANUAL, KSPROPERTY_CAMERACONTROL_S, KSPROPERTY_TYPE_GET, KSPROPERTY_TYPE_SET,
    PROPSETID_VIDCAP_CAMERACONTROL,
};
use windows::Win32::Media::MediaFoundation::{
    IMFActivate, IMFAttributes, IMFExtendedCameraControl, IMFExtendedCameraController, IMFGetService, IMFMediaSource, MF_CAPTURE_ENGINE_MEDIASOURCE, MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
    MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, MF_VERSION, MFCreateAttributes, MFEnumDeviceSources, MFSTARTUP_NOSOCKET, MFShutdown, MFStartup,
};
use windows::Win32::System::Com::{COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE, CoInitializeEx, CoTaskMemFree, CoUninitialize};
use windows::Win32::System::Registry::{HKEY, HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, KEY_READ, RRF_RT_REG_QWORD, RegCloseKey, RegEnumKeyExW, RegGetValueW, RegOpenKeyExW};
use windows::core::{GUID, HRESULT, HSTRING, Interface, PWSTR};

/// COM + Media Foundation for the calling thread, balanced on drop (both are reference counted).
struct Runtime {
    com: bool,
    mf: bool,
}

impl Runtime {
    fn start() -> Self {
        // Apartment threaded, as nokhwa initialises its camera thread; S_FALSE
        // (already initialised) still needs the matching uninitialise.
        let com = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE) }.is_ok();
        let mf = unsafe { MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET) }.is_ok();
        Self { com, mf }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe {
            if self.mf {
                let _ = MFShutdown();
            }
            if self.com {
                CoUninitialize();
            }
        }
    }
}

fn hr(error: &windows::core::Error) -> String {
    format!("{} (0x{:08X})", error.message().trim(), error.code().0 as u32)
}

/// Errors that mean "this control does not exist" rather than "could not read it".
fn is_absent(code: HRESULT) -> bool {
    matches!(
        code.0 as u32,
        0x8007_0490 // ERROR_NOT_FOUND / E_PROP_ID_UNSUPPORTED
            | 0x8007_0492 // ERROR_SET_NOT_FOUND / E_PROP_SET_UNSUPPORTED
            | 0x8000_4001 // E_NOTIMPL
            | 0x8007_0032 // ERROR_NOT_SUPPORTED
            | 0x8007_0001 // ERROR_INVALID_FUNCTION
            | 0x8000_4002 // E_NOINTERFACE
            | 0x8007_0057 // E_INVALIDARG (some drivers for unknown ids)
            | 0xC00D_36B9 // MF_E_NOT_FOUND
            | 0xC00D_36B1 // MF_E_ATTRIBUTENOTFOUND
            | 0xC00D_36BB // MF_E_UNSUPPORTED_SERVICE
    )
}

fn activates() -> Result<Vec<IMFActivate>, String> {
    unsafe {
        let mut attributes: Option<IMFAttributes> = None;
        MFCreateAttributes(&mut attributes, 1).map_err(|e| hr(&e))?;
        let attributes = attributes.ok_or("MFCreateAttributes returned nothing")?;
        attributes.SetGUID(&MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID).map_err(|e| hr(&e))?;
        let mut array: *mut Option<IMFActivate> = std::ptr::null_mut();
        let mut count = 0u32;
        MFEnumDeviceSources(&attributes, &mut array, &mut count).map_err(|e| hr(&e))?;
        let mut list = Vec::with_capacity(count as usize);
        if !array.is_null() {
            for i in 0..count as usize {
                if let Some(activate) = (*array.add(i)).take() {
                    list.push(activate);
                }
            }
            CoTaskMemFree(Some(array as *const c_void));
        }
        Ok(list)
    }
}

fn string_attribute(activate: &IMFActivate, key: &GUID) -> String {
    unsafe {
        let mut value = PWSTR::null();
        let mut length = 0u32;
        if activate.GetAllocatedString(key, &mut value, &mut length).is_err() || value.is_null() {
            return String::new();
        }
        let text = value.to_string().unwrap_or_default();
        CoTaskMemFree(Some(value.0 as *const c_void));
        text
    }
}

/// Capture devices in Media Foundation's order (the indices nokhwa uses); nothing is opened.
pub fn list_devices() -> Result<Vec<DeviceEntry>, String> {
    let _runtime = Runtime::start();
    Ok(activates()?
        .iter()
        .enumerate()
        .map(|(index, activate)| {
            let name = string_attribute(activate, &MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME);
            let id = string_attribute(activate, &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK);
            let virtual_camera = crate::is_virtual(&name, &id);
            DeviceEntry { index: index as u32, name, id, virtual_camera }
        })
        .collect())
}

/// The controls of one capture device, on a Media Foundation source of its own.
pub struct MfControls {
    ks: Option<IKsControl>,
    camera: Option<IAMCameraControl>,
    proc_amp: Option<IAMVideoProcAmp>,
    extended: Option<IMFExtendedCameraController>,
    /// Stream index the VFR control answered on.
    vfr_stream: Option<u32>,
    source: IMFMediaSource,
    activate: IMFActivate,
    id: String,
    // Last: COM / MF must outlive the interfaces above.
    _runtime: Runtime,
}

impl MfControls {
    /// Opens device `id` (symbolic link, case-insensitive), or the `index`-th device when `id` is empty or not found.
    pub fn open(index: u32, id: &str) -> Result<Self, String> {
        let runtime = Runtime::start();
        let list = activates()?;
        let by_id = (!id.is_empty()).then(|| list.iter().position(|a| string_attribute(a, &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK).eq_ignore_ascii_case(id))).flatten();
        let position = by_id.unwrap_or(index as usize);
        let activate = list.into_iter().nth(position).ok_or_else(|| format!("no capture device {position}"))?;
        let id = string_attribute(&activate, &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK);
        let source: IMFMediaSource = unsafe { activate.ActivateObject() }.map_err(|e| format!("opening the device: {}", hr(&e)))?;
        let ks = source.cast::<IKsControl>().ok();
        let camera = source.cast::<IAMCameraControl>().ok();
        let proc_amp = source.cast::<IAMVideoProcAmp>().ok();
        let extended = source.cast::<IMFGetService>().ok().and_then(|service| unsafe { service.GetService::<IMFExtendedCameraController>(&GUID::zeroed()) }.ok());
        Ok(Self { ks, camera, proc_amp, extended, vfr_stream: None, source, activate, id, _runtime: runtime })
    }

    /// `KSPROPERTY_VIDCAP_CAMERACONTROL` get (`value` None) or set, through IKsControl.
    fn ks_camera_control(&self, id: i32, value: Option<i32>) -> windows::core::Result<(i32, u32)> {
        let ks = self.ks.as_ref().ok_or_else(|| windows::core::Error::from(HRESULT(0x8000_4002u32 as i32)))?;
        let mut request = KSPROPERTY_CAMERACONTROL_S::default();
        request.Property.Anonymous.Anonymous = KSIDENTIFIER_0_0 { Set: PROPSETID_VIDCAP_CAMERACONTROL, Id: id as u32, Flags: if value.is_some() { KSPROPERTY_TYPE_SET } else { KSPROPERTY_TYPE_GET } };
        if let Some(value) = value {
            request.Value = value;
            request.Flags = KSPROPERTY_CAMERACONTROL_FLAGS_MANUAL as u32;
        }
        let mut data = request;
        let size = std::mem::size_of::<KSPROPERTY_CAMERACONTROL_S>() as u32;
        let mut returned = 0u32;
        unsafe { ks.KsProperty(&request.Property, size, &mut data as *mut _ as *mut c_void, size, &mut returned)? };
        Ok((data.Value, data.Flags))
    }

    fn vfr_control(&mut self) -> Option<IMFExtendedCameraControl> {
        let extended = self.extended.as_ref()?;
        let streams = match self.vfr_stream {
            Some(stream) => vec![stream],
            None => vec![0, MF_CAPTURE_ENGINE_MEDIASOURCE.0],
        };
        for stream in streams {
            if let Ok(control) = unsafe { extended.GetExtendedCameraControl(stream, KSPROPERTY_CAMERACONTROL_EXTENDED_VFR.0 as u32) }
                && unsafe { control.GetCapabilities() } != 0 {
                    self.vfr_stream = Some(stream);
                    return Some(control);
                }
        }
        None
    }

    fn range(&self, kind: ControlKind) -> windows::core::Result<Option<ControlInfo>> {
        let (mut min, mut max, mut step, mut default, mut caps, mut value, mut flags) = (0, 0, 0, 0, 0, 0, 0);
        let (name, auto_flag, manual_flag) = match kind {
            ControlKind::Exposure => {
                let Some(camera) = &self.camera else { return Ok(None) };
                unsafe {
                    camera.GetRange(CameraControl_Exposure.0, &mut min, &mut max, &mut step, &mut default, &mut caps)?;
                    camera.Get(CameraControl_Exposure.0, &mut value, &mut flags)?;
                }
                ("Exposure (IAMCameraControl, log2 s)", CameraControl_Flags_Auto.0, CameraControl_Flags_Manual.0)
            }
            ControlKind::Gain | ControlKind::Brightness | ControlKind::BacklightCompensation => {
                let Some(amp) = &self.proc_amp else { return Ok(None) };
                let (property, name) = match kind {
                    ControlKind::Gain => (VideoProcAmp_Gain.0, "Gain (IAMVideoProcAmp)"),
                    ControlKind::Brightness => (VideoProcAmp_Brightness.0, "Brightness (IAMVideoProcAmp)"),
                    _ => (VideoProcAmp_BacklightCompensation.0, "Backlight compensation (IAMVideoProcAmp)"),
                };
                unsafe {
                    amp.GetRange(property, &mut min, &mut max, &mut step, &mut default, &mut caps)?;
                    amp.Get(property, &mut value, &mut flags)?;
                }
                (name, VideoProcAmp_Flags_Auto.0, VideoProcAmp_Flags_Manual.0)
            }
            _ => return Ok(None),
        };
        Ok(Some(ControlInfo {
            kind,
            name: name.into(),
            min: min as i64,
            max: max as i64,
            step: step as i64,
            default: default as i64,
            value: value as i64,
            auto: flags & auto_flag != 0,
            supports_auto: caps & auto_flag != 0,
            // Drivers that leave the caps empty still accept manual values.
            supports_manual: caps & manual_flag != 0 || caps == 0,
        }))
    }
}

impl Drop for MfControls {
    fn drop(&mut self) {
        unsafe {
            let _ = self.source.Shutdown();
            let _ = self.activate.ShutdownObject();
        }
    }
}

impl CameraControls for MfControls {
    fn query(&mut self, kind: ControlKind) -> Result<Option<ControlInfo>, String> {
        let result = match kind {
            ControlKind::AutoExposurePriority => self.ks_camera_control(KSPROPERTY_CAMERACONTROL_AUTO_EXPOSURE_PRIORITY.0, None).map(|(value, _)| {
                Some(ControlInfo {
                    kind,
                    name: "Auto exposure priority (KSPROPERTY_CAMERACONTROL_AUTO_EXPOSURE_PRIORITY)".into(),
                    min: 0,
                    max: 1,
                    step: 1,
                    default: value as i64,
                    value: value as i64,
                    auto: false,
                    supports_auto: false,
                    supports_manual: true,
                })
            }),
            ControlKind::VariableFrameRate => {
                return Ok(self.vfr_control().map(|control| {
                    let flags = unsafe { control.GetFlags() } as i64;
                    ControlInfo { kind, name: "Variable frame rate (KSPROPERTY_CAMERACONTROL_EXTENDED_VFR)".into(), min: 0, max: 1, step: 1, default: flags, value: flags, auto: false, supports_auto: false, supports_manual: true }
                }));
            }
            _ => self.range(kind),
        };
        match result {
            Ok(info) => Ok(info),
            Err(e) if is_absent(e.code()) => Ok(None),
            Err(e) => Err(hr(&e)),
        }
    }

    fn set(&mut self, kind: ControlKind, value: ControlValue) -> Result<(), String> {
        let v = value.value.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        let result = match kind {
            ControlKind::AutoExposurePriority => self.ks_camera_control(KSPROPERTY_CAMERACONTROL_AUTO_EXPOSURE_PRIORITY.0, Some(v)).map(|_| ()),
            ControlKind::VariableFrameRate => {
                let control = self.vfr_control().ok_or("no variable frame rate control")?;
                unsafe { control.SetFlags(value.value as u64).and_then(|_| control.CommitSettings()) }
            }
            ControlKind::Exposure => {
                let camera = self.camera.as_ref().ok_or("no IAMCameraControl")?;
                let flags = if value.auto { CameraControl_Flags_Auto.0 } else { CameraControl_Flags_Manual.0 };
                unsafe { camera.Set(CameraControl_Exposure.0, v, flags) }
            }
            ControlKind::Gain | ControlKind::Brightness | ControlKind::BacklightCompensation => {
                let amp = self.proc_amp.as_ref().ok_or("no IAMVideoProcAmp")?;
                let property = match kind {
                    ControlKind::Gain => VideoProcAmp_Gain.0,
                    ControlKind::Brightness => VideoProcAmp_Brightness.0,
                    _ => VideoProcAmp_BacklightCompensation.0,
                };
                let flags = if value.auto { VideoProcAmp_Flags_Auto.0 } else { VideoProcAmp_Flags_Manual.0 };
                unsafe { amp.Set(property, v, flags) }
            }
        };
        result.map_err(|e| hr(&e))
    }

    fn exposure_units(&self) -> ExposureUnits {
        ExposureUnits::Log2Seconds
    }

    fn device_id(&self) -> String {
        self.id.clone()
    }
}

/// The app currently using a camera, from the privacy settings' usage
/// records (`LastUsedTimeStop == 0` while an app has a camera open).
pub fn camera_in_use() -> Option<String> {
    const STORE: &str = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam";
    for root in [HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE] {
        for path in [STORE.to_string(), format!(r"{STORE}\NonPackaged")] {
            for app in subkeys(root, &path) {
                let key = format!(r"{path}\{app}");
                let start = qword(root, &key, "LastUsedTimeStart");
                let stop = qword(root, &key, "LastUsedTimeStop");
                if start.is_some_and(|s| s != 0) && stop == Some(0) {
                    return Some(app.replace('#', r"\"));
                }
            }
        }
    }
    None
}

fn subkeys(root: HKEY, path: &str) -> Vec<String> {
    let mut names = Vec::new();
    unsafe {
        let mut key = HKEY::default();
        if RegOpenKeyExW(root, &HSTRING::from(path), None, KEY_READ, &mut key) != ERROR_SUCCESS {
            return names;
        }
        let mut buffer = vec![0u16; 1024];
        for index in 0.. {
            let mut length = buffer.len() as u32;
            if RegEnumKeyExW(key, index, Some(PWSTR(buffer.as_mut_ptr())), &mut length, None, None, None, None) != ERROR_SUCCESS {
                break;
            }
            names.push(String::from_utf16_lossy(&buffer[..length as usize]));
        }
        let _ = RegCloseKey(key);
    }
    names
}

fn qword(root: HKEY, path: &str, value: &str) -> Option<u64> {
    let mut data = 0u64;
    let mut size = std::mem::size_of::<u64>() as u32;
    let status = unsafe { RegGetValueW(root, &HSTRING::from(path), &HSTRING::from(value), RRF_RT_REG_QWORD, None, Some(&mut data as *mut u64 as *mut c_void), Some(&mut size)) };
    (status == ERROR_SUCCESS).then_some(data)
}

#[cfg(test)]
mod tests {
    #[test]
    fn listing_opens_nothing() {
        // Enumeration only reads attributes; it must work with any camera busy or none present.
        let devices = super::list_devices().expect("enumerate");
        for device in devices {
            assert!(!device.name.is_empty() || !device.id.is_empty());
        }
        let _ = super::camera_in_use();
    }
}
