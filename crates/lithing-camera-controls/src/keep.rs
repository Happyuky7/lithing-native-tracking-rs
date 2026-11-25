//! The "Keep frame rate" policy.
//!
//! 1. Probe: the first seconds run with the camera's own settings and give
//!    the delivered rate *before*.
//! 2. Primary fix (when the option is on): variable frame rate off (Windows
//!    extended control) and UVC auto-exposure priority = 0. Auto exposure
//!    stays on but may no longer stretch the frame period; the picture gets
//!    darker in a dim room instead of slower (the tracker's low light mode
//!    brightens it again).
//! 3. Fallback, only if the delivered rate stays under 80 % of the
//!    negotiated one for 3 s (no priority control, or a camera that ignores
//!    it): manual exposure at the longest value that fits the frame period,
//!    gain (else brightness) raised by the exposure it lost, then nudged by
//!    the measured frame luma. When the room is bright again the camera gets
//!    its automatic exposure back; if capping did not bring the rate back,
//!    it is undone and not retried.
//! 4. Restore: every control written is put back to the value it had before
//!    the first write (and a journal file covers crashes, see [`crate::journal`]).

use crate::journal::Journal;
use crate::{CameraControlReport, CameraControls, ControlChange, ControlInfo, ControlKind, ControlValue};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct KeepFrameRateConfig {
    /// Seconds measured with the camera's own settings before anything is changed.
    pub probe_secs: f64,
    /// Delivered / negotiated rate under which the rate counts as dropped.
    pub drop_ratio: f32,
    /// Consecutive dropped one-second windows before the exposure fallback.
    pub drop_windows: u32,
    /// Seconds after the fallback to judge whether it brought the rate back.
    pub verify_secs: f64,
    /// Seconds before the fallback may be tried again after the light came back.
    pub retry_after_secs: f64,
    /// Mean frame luma (0-255) band the gain is steered into during the fallback.
    pub luma_low: f32,
    pub luma_high: f32,
    /// Consecutive bright windows (at original gain) before automatic exposure comes back.
    pub bright_windows: u32,
    /// Restore journal folder (None: the temp directory); `use_journal` false disables it.
    pub journal_dir: Option<PathBuf>,
    pub use_journal: bool,
}

impl Default for KeepFrameRateConfig {
    fn default() -> Self {
        Self { probe_secs: 2.0, drop_ratio: 0.8, drop_windows: 3, verify_secs: 4.0, retry_after_secs: 30.0, luma_low: 70.0, luma_high: 165.0, bright_windows: 5, journal_dir: None, use_journal: true }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    Probing,
    Watching,
    Fallback { since: f64, fps_at_start: f32 },
}

pub struct KeepFrameRate {
    config: KeepFrameRateConfig,
    enabled: bool,
    target_fps: f32,
    phase: Phase,
    /// Values before the first write, in write order.
    originals: Vec<(ControlKind, ControlValue)>,
    device_id: String,
    journal: Option<Journal>,
    probe: Vec<f32>,
    low_windows: u32,
    bright_windows: u32,
    retry_at: f64,
    fallback_ineffective: bool,
    /// Gain or brightness, raised during the fallback, and its value before.
    boost: Option<(ControlKind, i64)>,
    primary_strategy: &'static str,
    pub report: CameraControlReport,
}

impl KeepFrameRate {
    pub fn new(device: &str, target_fps: f32, enabled: bool, config: KeepFrameRateConfig) -> Self {
        Self {
            config,
            enabled,
            target_fps: target_fps.max(1.0),
            phase: Phase::Probing,
            originals: Vec::new(),
            device_id: String::new(),
            journal: None,
            probe: Vec::new(),
            low_windows: 0,
            bright_windows: 0,
            retry_at: 0.0,
            fallback_ineffective: false,
            boost: None,
            primary_strategy: "none",
            report: CameraControlReport { device: device.into(), keep_frame_rate: enabled, target_fps, strategy: "none".into(), ..Default::default() },
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Records the controls the camera has and puts back values a crashed
    /// earlier run left changed. `controls` is None for a virtual camera or
    /// when they could not be opened (`error` says why).
    pub fn attach(&mut self, controls: Option<&mut dyn CameraControls>, error: Option<String>) {
        let Some(controls) = controls else {
            self.report.controls_error = error;
            return;
        };
        self.device_id = controls.device_id();
        if self.config.use_journal {
            let journal = Journal::for_device(self.config.journal_dir.as_deref(), &self.device_id);
            if let Some(left) = journal.load() {
                let mut failed = Vec::new();
                for (kind, value) in left.iter().rev() {
                    if let Err(e) = controls.set(*kind, *value) {
                        failed.push(format!("{}: {e}", kind.key()));
                    }
                }
                self.report.notes.push(format!("put back {} control value(s) an earlier run left changed", left.len()));
                if failed.is_empty() {
                    journal.clear();
                } else {
                    self.report.restore_errors.extend(failed);
                }
            }
            self.journal = Some(journal);
        }
        let (found, errors) = crate::query_all(controls);
        self.report.controls = found;
        if !errors.is_empty() {
            self.report.controls_error = Some(errors.join("; "));
        }
    }

    /// One measurement window (about a second): `now` in seconds since the
    /// stream opened, the delivered rate in it, the mean frame luma.
    pub fn window(&mut self, mut controls: Option<&mut dyn CameraControls>, now: f64, fps: f32, luma: Option<f32>) {
        self.report.luma = luma;
        match self.phase {
            Phase::Probing => {
                self.probe.push(fps);
                if now + 1e-6 >= self.config.probe_secs {
                    self.report.fps_before = Some(self.probe.iter().sum::<f32>() / self.probe.len() as f32);
                    self.phase = Phase::Watching;
                    if self.enabled {
                        if let Some(c) = controls.as_deref_mut() {
                            self.apply_primary(c, now);
                        } else {
                            self.report.strategy = "none".into();
                        }
                    } else {
                        self.report.strategy = "off".into();
                    }
                }
            }
            Phase::Watching => {
                self.report.fps_after = Some(fps);
                if !self.enabled {
                    return;
                }
                let Some(c) = controls else { return };
                if fps < self.config.drop_ratio * self.target_fps {
                    self.low_windows += 1;
                } else {
                    self.low_windows = 0;
                }
                if self.low_windows >= self.config.drop_windows && !self.fallback_ineffective && now >= self.retry_at {
                    self.low_windows = 0;
                    self.apply_fallback(c, now, fps);
                }
            }
            Phase::Fallback { since, fps_at_start } => {
                self.report.fps_after = Some(fps);
                let Some(c) = controls else { return };
                let dropped = fps < self.config.drop_ratio * self.target_fps;
                if now - since >= self.config.verify_secs && dropped && fps < fps_at_start * 1.15 {
                    self.leave_fallback(c, now);
                    self.fallback_ineffective = true;
                    self.report.notes.push(format!(
                        "capping the exposure did not bring the rate back ({fps:.1} of {:.0} fps; the camera may not deliver that rate in this format); automatic exposure restored, not retried",
                        self.target_fps
                    ));
                    return;
                }
                if let Some(luma) = luma {
                    self.steer_brightness(c, now, luma);
                }
            }
        }
    }

    /// Turns the option on or off while the camera runs: off puts every
    /// control back, on applies the fix again.
    pub fn set_enabled(&mut self, controls: Option<&mut dyn CameraControls>, enabled: bool, now: f64) {
        if self.enabled == enabled {
            return;
        }
        self.enabled = enabled;
        self.report.keep_frame_rate = enabled;
        if self.phase == Phase::Probing {
            return; // decided when the probe ends
        }
        let Some(c) = controls else { return };
        if enabled {
            self.report.restored = false;
            self.fallback_ineffective = false;
            self.low_windows = 0;
            self.phase = Phase::Watching;
            self.apply_primary(c, now);
        } else {
            self.restore(c);
            self.phase = Phase::Watching;
            self.report.strategy = "off".into();
        }
    }

    /// Puts every written control back to its value before the first write.
    pub fn restore(&mut self, controls: &mut dyn CameraControls) {
        if self.originals.is_empty() {
            return;
        }
        let mut errors = Vec::new();
        for (kind, value) in self.originals.iter().rev() {
            if let Err(e) = controls.set(*kind, *value) {
                errors.push(format!("{}: {e}", kind.key()));
            }
        }
        if errors.is_empty()
            && let Some(journal) = &self.journal {
                journal.clear();
            }
        self.report.restored = errors.is_empty();
        self.report.restore_errors.extend(errors);
        self.originals.clear();
        self.boost = None;
        if let Phase::Fallback { .. } = self.phase {
            self.phase = Phase::Watching;
        }
    }

    /// Whether any control is currently changed.
    pub fn changed(&self) -> bool {
        !self.originals.is_empty()
    }

    fn query(&mut self, c: &mut dyn CameraControls, kind: ControlKind) -> Option<ControlInfo> {
        match c.query(kind) {
            Ok(info) => info,
            Err(e) => {
                let note = format!("reading {}: {e}", kind.key());
                if !self.report.notes.contains(&note) {
                    self.report.notes.push(note);
                }
                None
            }
        }
    }

    fn write(&mut self, c: &mut dyn CameraControls, kind: ControlKind, to: ControlValue, reason: &str, now: f64) -> bool {
        let from = self.query(c, kind).map(|info| info.current());
        if !self.originals.iter().any(|(k, _)| *k == kind) {
            let Some(original) = from else {
                self.report.changes.push(ControlChange { kind, from, to, reason: reason.into(), error: Some("current value unreadable: not changed (it could not be restored)".into()), at: now });
                return false;
            };
            self.originals.push((kind, original));
            if let Some(journal) = &self.journal
                && let Err(e) = journal.save(&self.device_id, &self.originals) {
                    self.report.notes.push(format!("restore journal not written: {e}"));
                }
        }
        let result = c.set(kind, to);
        let ok = result.is_ok();
        self.report.changes.push(ControlChange { kind, from, to, reason: reason.into(), error: result.err(), at: now });
        ok
    }

    fn apply_primary(&mut self, c: &mut dyn CameraControls, now: f64) {
        let mut priority = false;
        if let Some(vfr) = self.query(c, ControlKind::VariableFrameRate) {
            if vfr.value != 0 {
                priority |= self.write(c, ControlKind::VariableFrameRate, ControlValue::manual(0), "variable frame rate off", now);
            } else {
                priority = true;
            }
        }
        if let Some(aep) = self.query(c, ControlKind::AutoExposurePriority) {
            if aep.value != 0 {
                priority |= self.write(c, ControlKind::AutoExposurePriority, ControlValue::manual(0), "auto exposure may not lower the frame rate", now);
            } else {
                priority = true;
                self.report.notes.push("auto-exposure priority was already off".into());
            }
        }
        let exposure = self.query(c, ControlKind::Exposure).is_some_and(|e| e.supports_manual);
        self.primary_strategy = if priority {
            "priority"
        } else if exposure {
            "exposure-cap-on-drop"
        } else {
            "unsupported"
        };
        self.report.strategy = self.primary_strategy.into();
    }

    fn apply_fallback(&mut self, c: &mut dyn CameraControls, now: f64, fps: f32) {
        let Some(exposure) = self.query(c, ControlKind::Exposure).filter(|e| e.supports_manual) else {
            self.fallback_ineffective = true;
            self.report.notes.push(format!("the rate dropped to {fps:.1} fps and the camera has no manual exposure to cap"));
            return;
        };
        let units = c.exposure_units();
        let period = 1.0 / self.target_fps as f64;
        let cap = units.longest_within(period, &exposure);
        // In auto mode the reported value may be stale; the dropped rate says
        // how long the camera was exposing at least.
        let current = if exposure.auto { units.seconds(exposure.value).max(1.0 / fps.max(1.0) as f64) } else { units.seconds(exposure.value) };
        let ratio = (current / units.seconds(cap)).clamp(1.0, 8.0);
        let capped = self.write(c, ControlKind::Exposure, ControlValue::manual(cap), &format!("frame rate dropped to {fps:.1}: manual exposure {:.1} ms fits {:.0} fps", units.seconds(cap) * 1000.0, self.target_fps), now);
        if !capped {
            self.leave_fallback(c, now);
            self.fallback_ineffective = true;
            return;
        }
        let boost = [ControlKind::Gain, ControlKind::Brightness].into_iter().find_map(|kind| self.query(c, kind).filter(|b| b.supports_manual && b.max > b.min));
        if let Some(b) = boost {
            self.boost = Some((b.kind, b.value));
            let base = (b.value - b.min).max((b.max - b.min) / 8) as f64;
            let wanted = b.align_down(b.min + (base * ratio).round() as i64).max(b.value);
            if wanted > b.value || b.auto {
                self.write(c, b.kind, ControlValue::manual(wanted), &format!("{} raised x{ratio:.1} for the shorter exposure", b.kind.key()), now);
            }
        } else {
            self.report.notes.push("no gain or brightness control: the picture gets darker (the tracker's low light mode can brighten it)".into());
        }
        self.phase = Phase::Fallback { since: now, fps_at_start: fps };
        self.bright_windows = 0;
        self.report.strategy = "exposure-cap".into();
    }

    fn steer_brightness(&mut self, c: &mut dyn CameraControls, now: f64, luma: f32) {
        let boost = match self.boost {
            Some((kind, original)) => self.query(c, kind).map(|info| (info, original)),
            None => None,
        };
        if luma > self.config.luma_high {
            if let Some((info, original)) = &boost
                && info.value > *original {
                    let delta = ((info.max - info.min) / 16).max(info.step).max(1);
                    let next = (info.value - delta).max(*original);
                    self.write(c, info.kind, ControlValue::manual(next), "bright again: less boost", now);
                    return;
                }
            self.bright_windows += 1;
            if self.bright_windows >= self.config.bright_windows {
                self.leave_fallback(c, now);
                self.report.notes.push(format!("light came back at {now:.0} s: automatic exposure restored"));
            }
            return;
        }
        self.bright_windows = 0;
        if luma < self.config.luma_low
            && let Some((info, _)) = boost
                && info.value < info.max {
                    let delta = ((info.max - info.min) / 16).max(info.step).max(1);
                    let next = info.align_down(info.value + delta).max(info.value + 1).min(info.max);
                    self.write(c, info.kind, ControlValue::manual(next), &format!("frame too dark (luma {luma:.0}): more boost"), now);
                }
    }

    /// Back to automatic exposure (and the original boost), keeping the primary fix.
    fn leave_fallback(&mut self, c: &mut dyn CameraControls, now: f64) {
        let mut put_back = Vec::new();
        for kind in [Some(ControlKind::Exposure), self.boost.map(|(kind, _)| kind)].into_iter().flatten() {
            if let Some((_, original)) = self.originals.iter().find(|(k, _)| *k == kind) {
                put_back.push((kind, *original));
            }
        }
        for (kind, original) in put_back {
            if let Err(e) = c.set(kind, original) {
                self.report.restore_errors.push(format!("{}: {e}", kind.key()));
            } else {
                self.report.changes.push(ControlChange { kind, from: None, to: original, reason: "back to the camera's own setting".into(), error: None, at: now });
            }
        }
        self.boost = None;
        self.bright_windows = 0;
        self.low_windows = 0;
        self.retry_at = now + self.config.retry_after_secs;
        self.phase = Phase::Watching;
        self.report.strategy = self.primary_strategy.into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExposureUnits;
    use std::collections::HashMap;

    /// A camera with a few controls and a crude model of its frame rate:
    /// in the dark, automatic exposure with priority on runs at `dark_fps`.
    struct Mock {
        controls: HashMap<ControlKind, ControlInfo>,
        units: ExposureUnits,
        fail_set: Vec<ControlKind>,
        writes: Vec<(ControlKind, ControlValue)>,
        dark: bool,
        dark_fps: f32,
        target: f32,
        /// The camera ignores the priority control.
        ignores_priority: bool,
    }

    fn control(kind: ControlKind, min: i64, max: i64, value: i64, auto: bool) -> ControlInfo {
        ControlInfo { kind, name: kind.key().into(), min, max, step: 1, default: value, value, auto, supports_auto: auto, supports_manual: true }
    }

    impl Mock {
        fn new(kinds: &[ControlKind]) -> Self {
            let mut controls = HashMap::new();
            for kind in kinds {
                let info = match kind {
                    ControlKind::AutoExposurePriority => control(*kind, 0, 1, 1, false),
                    ControlKind::VariableFrameRate => control(*kind, 0, 1, 1, false),
                    ControlKind::Exposure => control(*kind, -13, -1, -3, true),
                    ControlKind::Gain => control(*kind, 0, 100, 10, false),
                    ControlKind::Brightness => control(*kind, -64, 64, 0, false),
                    ControlKind::BacklightCompensation => control(*kind, 0, 1, 0, false),
                };
                controls.insert(*kind, info);
            }
            Self { controls, units: ExposureUnits::Log2Seconds, fail_set: Vec::new(), writes: Vec::new(), dark: true, dark_fps: 15.0, target: 30.0, ignores_priority: false }
        }

        fn value(&self, kind: ControlKind) -> Option<ControlValue> {
            self.controls.get(&kind).map(|c| c.current())
        }

        fn fps(&self) -> f32 {
            if !self.dark {
                return self.target;
            }
            let priority = self.value(ControlKind::AutoExposurePriority).is_some_and(|v| v.value != 0) || self.ignores_priority;
            let vfr = self.value(ControlKind::VariableFrameRate).is_some_and(|v| v.value != 0);
            let auto_exposure = self.value(ControlKind::Exposure).is_none_or(|v| v.auto);
            let has_priority = self.controls.contains_key(&ControlKind::AutoExposurePriority) || self.controls.contains_key(&ControlKind::VariableFrameRate);
            if auto_exposure && (priority || vfr || !has_priority) { self.dark_fps } else { self.target }
        }
    }

    impl CameraControls for Mock {
        fn query(&mut self, kind: ControlKind) -> Result<Option<ControlInfo>, String> {
            Ok(self.controls.get(&kind).cloned())
        }
        fn set(&mut self, kind: ControlKind, value: ControlValue) -> Result<(), String> {
            if self.fail_set.contains(&kind) {
                return Err("access denied".into());
            }
            let info = self.controls.get_mut(&kind).ok_or("missing")?;
            info.value = value.value;
            info.auto = value.auto;
            self.writes.push((kind, value));
            Ok(())
        }
        fn exposure_units(&self) -> ExposureUnits {
            self.units
        }
        fn device_id(&self) -> String {
            format!("mock-{:p}", self)
        }
    }

    fn config() -> KeepFrameRateConfig {
        KeepFrameRateConfig { use_journal: false, ..Default::default() }
    }

    /// Runs `seconds` one-second windows with the mock's rate; `luma` per window.
    fn run(keep: &mut KeepFrameRate, mock: &mut Mock, from: u32, to: u32, luma: impl Fn(&Mock) -> f32) {
        for second in from..to {
            let fps = mock.fps();
            let l = luma(mock);
            keep.window(Some(mock), (second + 1) as f64, fps, Some(l));
        }
    }

    #[test]
    fn priority_control_keeps_the_rate_and_is_restored() {
        let mut mock = Mock::new(&[ControlKind::AutoExposurePriority, ControlKind::Exposure, ControlKind::Gain]);
        let mut keep = KeepFrameRate::new("cam", 30.0, true, config());
        keep.attach(Some(&mut mock), None);
        assert_eq!(keep.report.controls.len(), 3);
        run(&mut keep, &mut mock, 0, 2, |_| 60.0);
        assert_eq!(keep.report.fps_before, Some(15.0));
        assert_eq!(keep.report.strategy, "priority");
        assert_eq!(mock.value(ControlKind::AutoExposurePriority), Some(ControlValue::manual(0)));
        run(&mut keep, &mut mock, 2, 10, |_| 60.0);
        assert_eq!(keep.report.fps_after, Some(30.0));
        // Exposure and gain were never touched.
        assert_eq!(mock.writes.len(), 1);
        keep.restore(&mut mock);
        assert_eq!(mock.value(ControlKind::AutoExposurePriority), Some(ControlValue::manual(1)));
        assert!(keep.report.restored);
        assert!(!keep.changed());
    }

    #[test]
    fn variable_frame_rate_is_turned_off_too() {
        let mut mock = Mock::new(&[ControlKind::VariableFrameRate, ControlKind::AutoExposurePriority]);
        let mut keep = KeepFrameRate::new("cam", 30.0, true, config());
        keep.attach(Some(&mut mock), None);
        run(&mut keep, &mut mock, 0, 6, |_| 90.0);
        assert_eq!(mock.value(ControlKind::VariableFrameRate).unwrap().value, 0);
        assert_eq!(mock.value(ControlKind::AutoExposurePriority).unwrap().value, 0);
        assert_eq!(keep.report.fps_after, Some(30.0));
        keep.restore(&mut mock);
        assert_eq!(mock.value(ControlKind::VariableFrameRate).unwrap().value, 1);
        assert_eq!(mock.value(ControlKind::AutoExposurePriority).unwrap().value, 1);
    }

    #[test]
    fn exposure_fallback_caps_raises_gain_and_hands_back_when_bright() {
        let mut mock = Mock::new(&[ControlKind::Exposure, ControlKind::Gain]);
        let mut keep = KeepFrameRate::new("cam", 30.0, true, config());
        keep.attach(Some(&mut mock), None);
        run(&mut keep, &mut mock, 0, 2, |_| 40.0);
        assert_eq!(keep.report.strategy, "exposure-cap-on-drop");
        assert!(mock.writes.is_empty(), "nothing is changed before the rate is seen to drop");
        // Three dropped windows -> the fallback.
        run(&mut keep, &mut mock, 2, 5, |_| 40.0);
        assert_eq!(keep.report.strategy, "exposure-cap");
        assert_eq!(mock.value(ControlKind::Exposure), Some(ControlValue::manual(-5)));
        let gain_after_cap = mock.value(ControlKind::Gain).unwrap().value;
        assert!(gain_after_cap > 10, "gain raised for the lost exposure ({gain_after_cap})");
        assert_eq!(mock.fps(), 30.0);
        // Still dark: the luma loop raises the gain further.
        run(&mut keep, &mut mock, 5, 8, |_| 40.0);
        let gain_dark = mock.value(ControlKind::Gain).unwrap().value;
        assert!(gain_dark > gain_after_cap);
        // Bright: the boost comes down to the original, then automatic exposure returns.
        mock.dark = false;
        run(&mut keep, &mut mock, 8, 40, |_| 200.0);
        assert_eq!(mock.value(ControlKind::Exposure), Some(ControlValue { value: -3, auto: true }));
        assert_eq!(mock.value(ControlKind::Gain).unwrap().value, 10);
        assert_eq!(keep.report.strategy, "exposure-cap-on-drop");
        // Dark again later (after the retry delay): the fallback comes back.
        mock.dark = true;
        run(&mut keep, &mut mock, 40, 80, |_| 40.0);
        assert_eq!(keep.report.strategy, "exposure-cap");
        keep.restore(&mut mock);
        assert_eq!(mock.value(ControlKind::Exposure), Some(ControlValue { value: -3, auto: true }));
        assert_eq!(mock.value(ControlKind::Gain), Some(ControlValue::manual(10)));
    }

    #[test]
    fn ineffective_fallback_is_undone_and_not_retried() {
        let mut mock = Mock::new(&[ControlKind::AutoExposurePriority, ControlKind::Exposure, ControlKind::Gain]);
        mock.ignores_priority = true;
        // The camera simply cannot do the rate: 15 fps whatever the exposure.
        let mut keep = KeepFrameRate::new("cam", 30.0, true, config());
        keep.attach(Some(&mut mock), None);
        for second in 0..30u32 {
            keep.window(Some(&mut mock), (second + 1) as f64, 15.0, Some(60.0));
        }
        assert!(keep.report.notes.iter().any(|n| n.contains("did not bring the rate back")));
        assert!(mock.value(ControlKind::Exposure).unwrap().auto);
        let exposure_writes = mock.writes.iter().filter(|(k, v)| *k == ControlKind::Exposure && !v.auto).count();
        assert_eq!(exposure_writes, 1, "tried once");
        keep.restore(&mut mock);
        assert_eq!(mock.value(ControlKind::AutoExposurePriority).unwrap().value, 1);
        assert_eq!(mock.value(ControlKind::Gain).unwrap().value, 10);
    }

    #[test]
    fn camera_without_controls_is_left_alone() {
        let mut mock = Mock::new(&[]);
        let mut keep = KeepFrameRate::new("OBS Virtual Camera", 30.0, true, config());
        keep.attach(Some(&mut mock), None);
        run(&mut keep, &mut mock, 0, 10, |_| 40.0);
        assert_eq!(keep.report.strategy, "unsupported");
        assert!(mock.writes.is_empty());
        // No controls object at all (virtual camera): only the rate is reported.
        let mut keep = KeepFrameRate::new("NVIDIA Broadcast", 30.0, true, config());
        keep.attach(None, Some("virtual camera".into()));
        for second in 0..5u32 {
            keep.window(None, (second + 1) as f64, 30.0, None);
        }
        assert_eq!(keep.report.fps_before, Some(30.0));
        assert_eq!(keep.report.fps_after, Some(30.0));
        assert_eq!(keep.report.strategy, "none");
    }

    #[test]
    fn turning_it_off_restores_and_on_reapplies() {
        let mut mock = Mock::new(&[ControlKind::AutoExposurePriority]);
        let mut keep = KeepFrameRate::new("cam", 30.0, true, config());
        keep.attach(Some(&mut mock), None);
        run(&mut keep, &mut mock, 0, 3, |_| 60.0);
        assert_eq!(mock.value(ControlKind::AutoExposurePriority).unwrap().value, 0);
        keep.set_enabled(Some(&mut mock), false, 3.5);
        assert_eq!(mock.value(ControlKind::AutoExposurePriority).unwrap().value, 1);
        assert_eq!(keep.report.strategy, "off");
        run(&mut keep, &mut mock, 3, 6, |_| 60.0);
        assert_eq!(mock.value(ControlKind::AutoExposurePriority).unwrap().value, 1, "nothing applied while off");
        keep.set_enabled(Some(&mut mock), true, 6.5);
        assert_eq!(mock.value(ControlKind::AutoExposurePriority).unwrap().value, 0);
        keep.restore(&mut mock);
        assert_eq!(mock.value(ControlKind::AutoExposurePriority).unwrap().value, 1);
    }

    #[test]
    fn off_from_the_start_changes_nothing() {
        let mut mock = Mock::new(&[ControlKind::AutoExposurePriority, ControlKind::Exposure, ControlKind::Gain]);
        let mut keep = KeepFrameRate::new("cam", 30.0, false, config());
        keep.attach(Some(&mut mock), None);
        run(&mut keep, &mut mock, 0, 20, |_| 40.0);
        assert!(mock.writes.is_empty());
        assert_eq!(keep.report.strategy, "off");
        assert_eq!(keep.report.fps_before, Some(15.0));
        assert_eq!(keep.report.fps_after, Some(15.0));
    }

    #[test]
    fn refused_writes_are_reported() {
        let mut mock = Mock::new(&[ControlKind::AutoExposurePriority]);
        mock.fail_set.push(ControlKind::AutoExposurePriority);
        let mut keep = KeepFrameRate::new("cam", 30.0, true, config());
        keep.attach(Some(&mut mock), None);
        run(&mut keep, &mut mock, 0, 3, |_| 60.0);
        assert_eq!(keep.report.changes.len(), 1);
        assert_eq!(keep.report.changes[0].error.as_deref(), Some("access denied"));
        assert_eq!(keep.report.strategy, "unsupported");
    }

    #[test]
    fn journal_puts_back_what_a_crashed_run_left() {
        let dir = std::env::temp_dir().join(format!("lithing-keep-journal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let journal_config = || KeepFrameRateConfig { journal_dir: Some(dir.clone()), use_journal: true, ..Default::default() };
        let mut mock = Mock::new(&[ControlKind::AutoExposurePriority, ControlKind::Exposure, ControlKind::Gain]);
        let mut keep = KeepFrameRate::new("cam", 30.0, true, journal_config());
        keep.attach(Some(&mut mock), None);
        run(&mut keep, &mut mock, 0, 3, |_| 60.0);
        assert_eq!(mock.value(ControlKind::AutoExposurePriority).unwrap().value, 0);
        // "Crash": the tracker is gone without restore().
        drop(keep);
        let mut next = KeepFrameRate::new("cam", 30.0, true, journal_config());
        next.attach(Some(&mut mock), None);
        assert_eq!(mock.value(ControlKind::AutoExposurePriority).unwrap().value, 1, "put back on the next open");
        assert!(next.report.notes.iter().any(|n| n.contains("earlier run")));
        // The values it reads as originals are the real ones.
        run(&mut next, &mut mock, 0, 3, |_| 60.0);
        next.restore(&mut mock);
        assert_eq!(mock.value(ControlKind::AutoExposurePriority).unwrap().value, 1);
        assert!(Journal::for_device(Some(&dir), &mock.device_id()).load().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
