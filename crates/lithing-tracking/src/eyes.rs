//! Port of src/app/eye-openness.ts: how closed each eye is, from the eyeBlink
//! blendshape, the lid gap over the eye width (3D) and the lid gap over the
//! iris diameter, each mapped between this user's open and closed level and
//! averaged by weights measured on labelled footage (normal light, a dark
//! room lit by the monitor, glasses). The levels come from the eye
//! calibration, or adapt: the open level follows the usual open eye, the
//! closed level is learnt from blinks. Keep the two files in step.

/// One cue's open and closed level for one eye, in the cue's own units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CueRange {
    pub open: f32,
    pub closed: f32,
}

/// One eye's calibrated levels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EyeRange {
    pub blink: CueRange,
    pub gap: CueRange,
    pub iris: CueRange,
}

/// The eye calibration (Calibrate eyes in the app): the user's own right and left eye.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct EyeCalibration {
    pub right: EyeRange,
    pub left: EyeRange,
    #[cfg_attr(feature = "serde", serde(default))]
    pub captured_at: f64,
}

/// One eye's raw cues of one face frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EyeCues {
    /// eyeBlink blendshape score of this eye.
    pub blink: f32,
    /// Lid gap over eye width (3D, aspect-corrected).
    pub gap: f32,
    /// Lid gap over iris diameter (0 without iris points).
    pub iris: f32,
}

/// Per eye, for the debug view and the eye calibration (sent with every face frame).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct EyeDebug {
    pub blink: f32,
    pub gap: f32,
    pub iris: f32,
    /// Geometric openness, 0 closed .. 1 open.
    pub geometry: f32,
    /// Fused closure before the curve.
    pub fused: f32,
    /// The closure sent on (after the curve and the calm filter).
    pub output: f32,
    pub gap_open: f32,
    pub gap_closed: f32,
    pub calibrated: bool,
}

/// Both eyes of the last face frame: the user's right, then left.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EyesDebug {
    pub right: EyeDebug,
    pub left: EyeDebug,
}

struct EyePoints {
    corners: [usize; 2],
    upper: [usize; 3],
    lower: [usize; 3],
    ring: [usize; 4],
}

const EYE_POINTS: [EyePoints; 2] = [
    // The user's right eye (left in the camera image).
    EyePoints { corners: [33, 133], upper: [160, 159, 158], lower: [144, 145, 153], ring: [469, 470, 471, 472] },
    EyePoints { corners: [263, 362], upper: [387, 386, 385], lower: [373, 374, 380], ring: [474, 475, 476, 477] },
];

/// The geometric cues of one eye (0 the user's right, 1 the left); `aspect` is height / width.
/// The blink cue is left at 0 for the caller to fill.
pub fn measure_eye_geometry(points: &[[f32; 3]], eye: usize, aspect: f32) -> EyeCues {
    let spec = &EYE_POINTS[eye];
    let distance = |a: usize, b: usize| {
        let (p, q) = (points[a], points[b]);
        ((p[0] - q[0]).powi(2) + ((p[1] - q[1]) * aspect).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    };
    let width = distance(spec.corners[0], spec.corners[1]).max(1e-4);
    let gap = (0..3).map(|i| distance(spec.upper[i], spec.lower[i])).sum::<f32>() / 3.0;
    let mut cues = EyeCues { blink: 0.0, gap: gap / width, iris: 0.0 };
    if points.len() > spec.ring[3] {
        let flat = |a: usize, b: usize| (points[a][0] - points[b][0]).hypot((points[a][1] - points[b][1]) * aspect);
        let iris = (flat(spec.ring[0], spec.ring[2]) + flat(spec.ring[1], spec.ring[3])) / 2.0;
        let flat_gap = (0..3).map(|i| flat(spec.upper[i], spec.lower[i])).sum::<f32>() / 3.0;
        cues.iris = if iris > 1e-4 { flat_gap / iris } else { 0.0 };
    }
    cues
}

/// Tuning, the same numbers as EYE_TUNING in eye-openness.ts.
pub struct Tuning {
    pub prior: [f32; 3],
    pub initial_span: [f32; 3],
    /// The same in low light mode, where a closed eye's cues barely move.
    pub initial_span_dark: [f32; 3],
    pub min_span: [f32; 3],
    pub weights_plain: [f32; 3],
    pub weights_glasses: [f32; 3],
    pub weights_dark: [f32; 3],
    pub open_quantile: f32,
    pub open_rate: f32,
    pub open_gate: f32,
    /// Way out of a lock when the light drops (see recoverAfter in eye-openness.ts).
    pub recover_after: f32,
    pub recover_time: f32,
    pub closed_attack: f32,
    pub closed_relax: f32,
    pub closed_gate: f32,
    pub calibrated_reach: f32,
    pub curve_dead_zone: f32,
    pub curve_full: f32,
    pub calm_band: f32,
    pub calm_time: f32,
    pub calm_band_dark: f32,
    pub calm_time_dark: f32,
}

// Cue order everywhere: blink, gap, iris.
pub const TUNING: Tuning = Tuning {
    prior: [0.12, 0.27, 0.62],
    initial_span: [0.16, 0.4, 0.4],
    initial_span_dark: [0.2, 0.4, 0.24],
    min_span: [0.14, 0.3, 0.24],
    // Glasses did not hurt the geometric cues more than the blendshape on the
    // labelled footage, so glasses mode keeps the plain weights; in a dark room
    // the iris-scaled gap holds best.
    weights_plain: [1.0, 1.4, 1.0],
    weights_glasses: [1.0, 1.4, 1.0],
    weights_dark: [1.0, 0.7, 1.4],
    open_quantile: 0.8,
    open_rate: 0.5,
    open_gate: 0.25,
    recover_after: 4.0,
    recover_time: 3.0,
    closed_attack: 2.0,
    closed_relax: 0.1,
    closed_gate: 0.8,
    calibrated_reach: 0.6,
    curve_dead_zone: 0.2,
    curve_full: 0.83,
    calm_band: 0.14,
    calm_time: 0.15,
    calm_band_dark: 0.25,
    calm_time_dark: 0.2,
};

/// The "Eye openness" response curve: `sensitivity` 0..1, 0.5 default.
pub fn eye_curve(closure: f32, sensitivity: f32) -> f32 {
    let shift = 0.5 - sensitivity.clamp(0.0, 1.0);
    let dead_zone = (TUNING.curve_dead_zone + shift * 0.16).max(0.02);
    let full = TUNING.curve_full + shift * 0.24;
    ((closure - dead_zone) / (full - dead_zone)).clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Debug, Default)]
struct CueState {
    open: f32,
    closed: f32,
    calibrated: bool,
}

#[derive(Clone, Copy, Debug)]
struct EyeState {
    cues: [CueState; 3],
    calm: f32,
    /// Seconds the eye has read not open (leaks away while it reads open).
    shut: f32,
}

impl EyeState {
    fn new(span: [f32; 3]) -> Self {
        let t = &TUNING;
        Self {
            cues: [
                CueState { open: t.prior[0], closed: t.prior[0] + span[0], calibrated: false },
                CueState { open: t.prior[1], closed: t.prior[1] * (1.0 - span[1]), calibrated: false },
                CueState { open: t.prior[2], closed: t.prior[2] * (1.0 - span[2]), calibrated: false },
            ],
            calm: 0.0,
            shut: 0.0,
        }
    }
}

fn closure_of(value: f32, state: &CueState) -> f32 {
    let span = state.closed - state.open;
    if span.abs() < 1e-6 { 0.0 } else { ((value - state.open) / span).clamp(-0.25, 1.25) }
}

/// Moves one cue's levels with this frame (see `adapt` in eye-openness.ts).
fn adapt(state: &mut CueState, value: f32, others: f32, dt: f32, initial_span: f32, min_span: f32, relative: bool, recover: f32) {
    let t = &TUNING;
    let direction = if state.closed > state.open { 1.0 } else { -1.0 };
    let span = (state.closed - state.open).abs();
    if others < t.open_gate {
        let step = t.open_rate * span * dt * if state.calibrated { 0.25 } else { 1.0 };
        let more_open = (state.open - value) * direction > 0.0;
        state.open += direction * if more_open { -step * t.open_quantile } else { step * (1.0 - t.open_quantile) };
    }
    if recover > 0.0 {
        state.open += (value - state.open) * recover;
    }
    let beyond = (value - state.closed) * direction > 0.0;
    if beyond && others > t.closed_gate {
        state.closed += (value - state.closed) * (1.0 - (-t.closed_attack * dt).exp());
    } else if !state.calibrated {
        let relaxed = state.open + direction * if relative { state.open.abs() * initial_span } else { initial_span };
        if (state.closed - relaxed) * direction > 0.0 {
            state.closed += (relaxed - state.closed) * (1.0 - (-t.closed_relax * dt).exp());
        }
    }
    let least = if relative { state.open.abs() * min_span } else { min_span };
    if (state.closed - state.open) * direction < least {
        state.closed = state.open + direction * least;
    }
}

/// The estimator, one per face tracker.
pub struct EyeOpenness {
    eyes: [EyeState; 2],
    calibration: Option<EyeCalibration>,
    pub sensitivity: f32,
    pub glasses: bool,
    pub low_light: bool,
    last_time: f64,
    pub debug: EyesDebug,
}

impl Default for EyeOpenness {
    fn default() -> Self {
        Self { eyes: [EyeState::new(TUNING.initial_span); 2], calibration: None, sensitivity: 0.5, glasses: false, low_light: false, last_time: f64::NEG_INFINITY, debug: EyesDebug::default() }
    }
}

impl EyeOpenness {
    pub fn calibration(&self) -> Option<&EyeCalibration> {
        self.calibration.as_ref()
    }

    pub fn set_calibration(&mut self, calibration: Option<EyeCalibration>) {
        self.calibration = calibration;
        self.reset();
    }

    fn initial_span(&self) -> [f32; 3] {
        if self.low_light { TUNING.initial_span_dark } else { TUNING.initial_span }
    }

    /// Back to the calibrated (or default) levels, as on a new camera start.
    pub fn reset(&mut self) {
        for eye in 0..2 {
            let mut state = EyeState::new(self.initial_span());
            if let Some(calibration) = &self.calibration {
                let range = if eye == 0 { calibration.right } else { calibration.left };
                for (cue, r) in state.cues.iter_mut().zip([range.blink, range.gap, range.iris]) {
                    // Eyes shut on purpose for the calibration close further than in a blink.
                    *cue = CueState { open: r.open, closed: r.open + (r.closed - r.open) * TUNING.calibrated_reach, calibrated: true };
                }
            }
            self.eyes[eye] = state;
        }
        self.last_time = f64::NEG_INFINITY;
    }

    /// One face frame at `time` (ms): returns the closure of the user's right and left eye.
    pub fn update(&mut self, right: EyeCues, left: EyeCues, time: f64) -> (f32, f32) {
        let t = &TUNING;
        let first = !self.last_time.is_finite();
        let dt = if first { 1.0 / 30.0 } else { ((time - self.last_time) / 1000.0).clamp(0.001, 0.25) as f32 };
        self.last_time = time;
        let weights = if self.low_light { t.weights_dark } else if self.glasses { t.weights_glasses } else { t.weights_plain };
        let spans = self.initial_span();
        let mut out = [0.0f32; 2];
        for eye in 0..2 {
            let cues = if eye == 0 { right } else { left };
            let state = &mut self.eyes[eye];
            let has_iris = cues.iris > 0.0;
            let blink_closure = closure_of(cues.blink, &state.cues[0]);
            let gap_closure = closure_of(cues.gap, &state.cues[1]);
            let iris_closure = if has_iris { closure_of(cues.iris, &state.cues[2]) } else { 0.0 };
            let iris_weight = if has_iris { weights[2] } else { 0.0 };
            let fused = ((blink_closure * weights[0] + gap_closure * weights[1] + iris_closure * iris_weight) / (weights[0] + weights[1] + iris_weight)).clamp(0.0, 1.0);
            let geometric = if has_iris { (gap_closure * weights[1] + iris_closure * weights[2]) / (weights[1] + weights[2]) } else { gap_closure };
            state.shut = if fused < t.open_gate { (state.shut - 2.0 * dt).max(0.0) } else { state.shut + dt };
            let recover = if state.shut > t.recover_after { 1.0 - (-dt / t.recover_time).exp() } else { 0.0 };
            adapt(&mut state.cues[0], cues.blink, geometric, dt, spans[0], t.min_span[0], false, recover);
            adapt(&mut state.cues[1], cues.gap, blink_closure, dt, spans[1], t.min_span[1], true, recover);
            if has_iris {
                adapt(&mut state.cues[2], cues.iris, blink_closure.max(gap_closure), dt, spans[2], t.min_span[2], true, recover);
            }
            let shaped = eye_curve(fused, self.sensitivity);
            let change = shaped - state.calm;
            let (calm_band, calm_time) = if self.low_light { (t.calm_band_dark, t.calm_time_dark) } else { (t.calm_band, t.calm_time) };
            if first || change.abs() >= calm_band {
                state.calm = shaped;
            } else {
                let gain = 1.0 - (-dt / calm_time * (change.abs() / calm_band).max(0.25).powi(2)).exp();
                state.calm += change * gain.min(1.0);
            }
            out[eye] = state.calm;
            let debug = if eye == 0 { &mut self.debug.right } else { &mut self.debug.left };
            *debug = EyeDebug {
                blink: cues.blink,
                gap: cues.gap,
                iris: cues.iris,
                geometry: 1.0 - gap_closure.clamp(0.0, 1.0),
                fused,
                output: state.calm,
                gap_open: state.cues[1].open,
                gap_closed: state.cues[1].closed,
                calibrated: state.cues[1].calibrated,
            };
        }
        (out[0], out[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cues(blink: f32, gap: f32, iris: f32) -> EyeCues {
        EyeCues { blink, gap, iris }
    }

    #[test]
    fn open_eyes_stay_open_and_a_blink_closes() {
        let mut eyes = EyeOpenness::default();
        let mut time = 0.0;
        for _ in 0..90 {
            let (r, l) = eyes.update(cues(0.08, 0.29, 0.66), cues(0.08, 0.29, 0.66), time);
            assert!(r < 0.05 && l < 0.05, "open eye droops: {r} {l}");
            time += 33.3;
        }
        let mut closed = 0.0f32;
        for _ in 0..4 {
            closed = eyes.update(cues(0.62, 0.05, 0.1), cues(0.62, 0.05, 0.1), time).0;
            time += 33.3;
        }
        assert!(closed > 0.95, "blink closes {closed}");
    }

    #[test]
    fn half_closed_is_in_between() {
        let mut eyes = EyeOpenness::default();
        let mut time = 0.0;
        for _ in 0..60 {
            eyes.update(cues(0.08, 0.29, 0.66), cues(0.08, 0.29, 0.66), time);
            time += 33.3;
        }
        // A few blinks, a few seconds apart, teach the closed level.
        for _ in 0..4 {
            for _ in 0..5 {
                eyes.update(cues(0.62, 0.05, 0.1), cues(0.62, 0.05, 0.1), time);
                time += 33.3;
            }
            for _ in 0..60 {
                eyes.update(cues(0.08, 0.29, 0.66), cues(0.08, 0.29, 0.66), time);
                time += 33.3;
            }
        }
        let mut half = (0.0, 0.0);
        for _ in 0..30 {
            half = eyes.update(cues(0.3, 0.17, 0.4), cues(0.3, 0.17, 0.4), time);
            time += 33.3;
        }
        assert!(half.0 > 0.2 && half.0 < 0.85, "half-closed eye: {}", half.0);
    }

    #[test]
    fn calibration_sets_the_levels() {
        let range = EyeRange { blink: CueRange { open: 0.06, closed: 0.3 }, gap: CueRange { open: 0.28, closed: 0.19 }, iris: CueRange { open: 0.7, closed: 0.52 } };
        let mut eyes = EyeOpenness::default();
        eyes.set_calibration(Some(EyeCalibration { right: range, left: range, captured_at: 0.0 }));
        // A dark room's shallow blink reads as closed with this user's levels.
        let (r, _) = eyes.update(cues(0.3, 0.19, 0.52), cues(0.3, 0.19, 0.52), 0.0);
        assert!(r > 0.95, "{r}");
        assert!(eyes.debug.right.calibrated);
    }

    #[test]
    fn eyes_are_independent() {
        let mut eyes = EyeOpenness::default();
        let mut time = 0.0;
        let mut wink = (0.0, 0.0);
        for _ in 0..40 {
            wink = eyes.update(cues(0.08, 0.29, 0.66), cues(0.65, 0.04, 0.08), time);
            time += 33.3;
        }
        assert!(wink.0 < 0.05 && wink.1 > 0.95, "wink {wink:?}");
    }

    #[test]
    fn a_darker_room_does_not_lock_the_eyes_shut() {
        let mut eyes = EyeOpenness::default();
        let mut time = 0.0;
        for _ in 0..90 {
            eyes.update(cues(0.08, 0.29, 0.66), cues(0.08, 0.29, 0.66), time);
            time += 33.3;
        }
        // The light drops: the same open eye now reads a little closed on every cue.
        let dark = cues(0.22, 0.2, 0.47);
        let mut out = (0.0, 0.0);
        for _ in 0..(15 * 30) {
            out = eyes.update(dark, dark, time);
            time += 33.3;
        }
        assert!(out.0 < 0.1 && out.1 < 0.1, "eyes still shut in the dark: {out:?}");
        // And a blink still closes in the dark.
        let mut closed = 0.0f32;
        for _ in 0..5 {
            closed = eyes.update(cues(0.5, 0.06, 0.12), cues(0.5, 0.06, 0.12), time).0;
            time += 33.3;
        }
        assert!(closed > 0.9, "blink in the dark: {closed}");
    }

    #[test]
    fn eyes_closed_for_a_few_seconds_stay_closed() {
        let mut eyes = EyeOpenness::default();
        let mut time = 0.0;
        for _ in 0..90 {
            eyes.update(cues(0.08, 0.29, 0.66), cues(0.08, 0.29, 0.66), time);
            time += 33.3;
        }
        let mut closed = 0.0f32;
        for _ in 0..(4 * 30) {
            closed = eyes.update(cues(0.62, 0.05, 0.1), cues(0.62, 0.05, 0.1), time).0;
            time += 33.3;
        }
        assert!(closed > 0.95, "held close: {closed}");
    }
}
