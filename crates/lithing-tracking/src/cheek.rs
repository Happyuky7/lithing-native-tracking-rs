//! Port of src/app/cheek-detector.ts: cheek puff from the face mesh
//! geometry (lower cheek width over the eye span, with closed lips, no smile
//! and a mouth no wider than at rest), fused with MediaPipe's cheekPuff,
//! which barely moves. See the web file for the rationale.

/// Cheek puff calibration: the raw width with the cheeks relaxed and puffed
/// (0: not captured; the relaxed width is then learned and a puff assumed
/// [`DEFAULT_PUFF`] wider).
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, rename_all = "camelCase"))]
pub struct CheekTuning {
    pub sensitivity: f32,
    pub relaxed_level: f32,
    pub puffed_level: f32,
}

impl Default for CheekTuning {
    fn default() -> Self {
        Self { sensitivity: 1.0, relaxed_level: 0.0, puffed_level: 0.0 }
    }
}

/// Cheek detector reading.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct CheekReading {
    /// 0-1: the larger of MediaPipe's cheekPuff and the geometric cue.
    pub puff: f32,
    /// 0-1: the geometric cue alone, after the gates.
    pub geometric: f32,
    /// Lower cheek width over the eye span (uncalibrated).
    pub raw: f32,
    /// The relaxed width in use (captured or learned).
    pub relaxed: f32,
    pub valid: bool,
}

pub const DEFAULT_PUFF: f32 = 0.08;
const CHEEK_PAIRS: [(usize, usize); 4] = [(187, 411), (147, 376), (207, 427), (192, 416)];
const EYE_OUTER: (usize, usize) = (33, 263);

#[derive(Default)]
pub struct CheekDetector {
    pub reading: CheekReading,
    learned_relaxed: f32,
    rest_mouth: f32,
    smoothed: f32,
}

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

impl CheekDetector {
    pub fn reset(&mut self) {
        self.smoothed = 0.0;
        self.reading.puff = 0.0;
        self.reading.geometric = 0.0;
        self.reading.valid = false;
    }

    /// `points`: the face mesh (normalized) of a `width` x `height` frame.
    /// `cheek_puff`, `smile` (mean of both sides) and `jaw_open`: MediaPipe's scores of the same face.
    #[allow(clippy::too_many_arguments)]
    pub fn update(&mut self, points: &[[f32; 3]], width: f32, height: f32, cheek_puff: f32, smile: f32, jaw_open: f32, tuning: &CheekTuning) -> CheekReading {
        if points.len() < 468 || width <= 0.0 || height <= 0.0 {
            self.reading.valid = false;
            return self.reading;
        }
        let distance = |a: usize, b: usize| ((points[a][0] - points[b][0]) * width).hypot((points[a][1] - points[b][1]) * height);
        let eyes = distance(EYE_OUTER.0, EYE_OUTER.1);
        if eyes < 8.0 {
            self.reading.valid = false;
            return self.reading;
        }
        let raw = CHEEK_PAIRS.iter().map(|&(a, b)| distance(a, b)).sum::<f32>() / CHEEK_PAIRS.len() as f32 / eyes;
        let mouth = distance(61, 291) / eyes;
        let lips_apart = distance(13, 14) / distance(61, 291).max(1.0);
        // At rest the widths are learned as slow floors, so a held puff does not become the rest.
        let resting = lips_apart < 0.06 && smile < 0.2 && jaw_open < 0.12;
        if self.learned_relaxed == 0.0 {
            self.learned_relaxed = raw;
            self.rest_mouth = mouth;
        } else if resting {
            self.learned_relaxed += (raw - self.learned_relaxed) * if raw < self.learned_relaxed { 0.03 } else { 0.004 };
            self.rest_mouth += (mouth - self.rest_mouth) * if mouth < self.rest_mouth { 0.2 } else { 0.01 };
        }
        let relaxed = if tuning.relaxed_level > 0.0 { tuning.relaxed_level } else { self.learned_relaxed };
        let puffed = if tuning.relaxed_level > 0.0 && tuning.puffed_level > tuning.relaxed_level { tuning.puffed_level } else { relaxed * (1.0 + DEFAULT_PUFF) };
        // Gates: puffed cheeks hold the lips shut and do not smile or widen the mouth.
        let closed = clamp01((0.08 - lips_apart) / 0.05) * clamp01((0.2 - jaw_open) / 0.1);
        let no_smile = clamp01((0.3 - smile) / 0.2);
        let narrow = clamp01((1.06 - mouth / self.rest_mouth.max(1e-3)) / 0.05);
        // A small dead zone keeps landmark jitter at rest out.
        let level = clamp01(((raw - relaxed) / (puffed - relaxed).max(1e-3) - 0.25) / 0.75 * tuning.sensitivity);
        let geometric = level * closed * no_smile * narrow;
        self.smoothed += (geometric - self.smoothed) * 0.5;
        self.reading = CheekReading { puff: cheek_puff.max(self.smoothed), geometric: self.smoothed, raw, relaxed, valid: true };
        self.reading
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A face mesh with the lower cheeks `cheeks` apart and the eye corners 1.0 apart (frame 100 x 100).
    fn face(cheeks: f32, lips: f32) -> Vec<[f32; 3]> {
        let mut points = vec![[0.5, 0.5, 0.0]; 478];
        points[33] = [0.0, 0.3, 0.0];
        points[263] = [1.0, 0.3, 0.0];
        for (a, b) in CHEEK_PAIRS {
            points[a] = [0.5 - cheeks / 2.0, 0.6, 0.0];
            points[b] = [0.5 + cheeks / 2.0, 0.6, 0.0];
        }
        points[61] = [0.25, 0.7, 0.0];
        points[291] = [0.75, 0.7, 0.0];
        points[13] = [0.5, 0.7, 0.0];
        points[14] = [0.5, 0.7 + lips, 0.0];
        points
    }

    #[allow(clippy::too_many_arguments)]
    fn settle(detector: &mut CheekDetector, cheeks: f32, lips: f32, smile: f32, jaw: f32, tuning: &CheekTuning, frames: usize) {
        for _ in 0..frames {
            detector.update(&face(cheeks, lips), 100.0, 100.0, 0.0, smile, jaw, tuning);
        }
    }

    #[test]
    fn puffed_cheeks_with_closed_lips_read_as_a_puff() {
        let tuning = CheekTuning::default();
        let mut detector = CheekDetector::default();
        settle(&mut detector, 1.0, 0.0, 0.0, 0.0, &tuning, 30);
        assert!(detector.reading.geometric < 0.05);
        settle(&mut detector, 1.1, 0.0, 0.0, 0.0, &tuning, 10);
        assert!(detector.reading.puff > 0.8, "{:?}", detector.reading);
        // The same width while smiling, or with the lips apart, is no puff.
        let mut other = CheekDetector::default();
        settle(&mut other, 1.0, 0.0, 0.0, 0.0, &tuning, 30);
        settle(&mut other, 1.1, 0.0, 0.5, 0.0, &tuning, 10);
        assert!(other.reading.puff < 0.05, "{:?}", other.reading);
        settle(&mut other, 1.1, 0.2, 0.0, 0.3, &tuning, 10);
        assert!(other.reading.puff < 0.05, "{:?}", other.reading);
    }

    #[test]
    fn captured_levels_set_the_range() {
        let tuning = CheekTuning { sensitivity: 1.0, relaxed_level: 1.0, puffed_level: 1.2 };
        let mut detector = CheekDetector::default();
        settle(&mut detector, 1.1, 0.0, 0.0, 0.0, &tuning, 20);
        // Halfway to the captured puff, past the dead zone.
        assert!((detector.reading.geometric - (0.5 - 0.25) / 0.75).abs() < 0.02, "{:?}", detector.reading);
    }
}
