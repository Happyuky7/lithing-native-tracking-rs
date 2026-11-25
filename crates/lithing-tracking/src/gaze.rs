//! Port of src/app/gaze.ts: the pupil (gaze) calibration. The raw gaze
//! (`eye_x` / `eye_y` in face_post.rs, before the baseline and mirroring) is
//! captured while the user looks at the centre and at each edge of the
//! screen; each side is then scaled on its own so that looking at an edge
//! reaches the full -1..1. Without it the gaze is only offset by the forward
//! baseline and most users never get past a third of the range, so the
//! pupils barely move. Keep the two files in step.

/// Raw gaze while looking at the centre of the screen.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct GazeCenter {
    pub x: f32,
    pub y: f32,
}

/// Raw gaze (raw eye_x / eye_y: before baseline and mirroring) looking at
/// the centre and each edge of the screen (Calibrate pupils in the app).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct GazeCalibration {
    pub center: GazeCenter,
    /// Raw eye_x looking at the left / right edge.
    pub left: f32,
    pub right: f32,
    /// Raw eye_y looking at the top / bottom edge.
    pub up: f32,
    pub down: f32,
    #[cfg_attr(feature = "serde", serde(default))]
    pub captured_at: f64,
}

/// One frame's raw gaze (raw eye_x / eye_y), for the debug view and the
/// pupil calibration capture (sent with every face frame).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GazeSample {
    pub x: f32,
    pub y: f32,
}

/// A side whose captured edge is closer to the centre than this was not
/// really measured (the user did not move the eyes): that side keeps the
/// plain offset rather than blowing noise up to the full range.
pub const GAZE_MIN_SPAN: f32 = 0.04;

/// One axis: the offset from `center`, over the span towards whichever of
/// `a` / `b` lies on that side (so the order of the two edges does not
/// matter), clamped to -1..1. A side narrower than [`GAZE_MIN_SPAN`] falls
/// back to the plain offset.
pub fn map_gaze_axis(raw: f32, center: f32, a: f32, b: f32) -> f32 {
    let hi = a.max(b);
    let lo = a.min(b);
    let d = raw - center;
    let mapped = if d >= 0.0 {
        let span = hi - center;
        if span >= GAZE_MIN_SPAN { d / span } else { d }
    } else {
        let span = center - lo;
        if span >= GAZE_MIN_SPAN { d / span } else { d }
    };
    mapped.clamp(-1.0, 1.0)
}

/// Both axes: x from (center.x, left, right), y from (center.y, up, down).
pub fn map_gaze(raw_x: f32, raw_y: f32, c: &GazeCalibration) -> (f32, f32) {
    (map_gaze_axis(raw_x, c.center.x, c.left, c.right), map_gaze_axis(raw_y, c.center.y, c.up, c.down))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn each_side_scales_on_its_own() {
        // Centre 0.1, edges at -0.2 and 0.3: 0.3 below, 0.2 above.
        assert!(close(map_gaze_axis(0.1, 0.1, -0.2, 0.3), 0.0));
        assert!(close(map_gaze_axis(0.2, 0.1, -0.2, 0.3), 0.5));
        assert!(close(map_gaze_axis(0.3, 0.1, -0.2, 0.3), 1.0));
        assert!(close(map_gaze_axis(-0.05, 0.1, -0.2, 0.3), -0.5));
        assert!(close(map_gaze_axis(-0.2, 0.1, -0.2, 0.3), -1.0));
    }

    #[test]
    fn edge_order_does_not_flip_the_sign() {
        for raw in [-0.3, -0.1, 0.0, 0.05, 0.2, 0.4] {
            assert_eq!(map_gaze_axis(raw, 0.0, -0.25, 0.2), map_gaze_axis(raw, 0.0, 0.2, -0.25));
        }
        assert!(map_gaze_axis(0.1, 0.0, 0.2, -0.25) > 0.0);
        assert!(map_gaze_axis(-0.1, 0.0, 0.2, -0.25) < 0.0);
    }

    #[test]
    fn narrow_sides_keep_the_plain_offset() {
        // Upper edge only 0.03 from the centre: d stays d.
        assert!(close(map_gaze_axis(0.12, 0.1, -0.2, 0.13), 0.02));
        // Lower edge at the centre (both edges on one side): d stays d.
        assert!(close(map_gaze_axis(0.05, 0.1, 0.1, 0.4), -0.05));
        // Exactly the minimum span still scales.
        assert!(close(map_gaze_axis(0.12, 0.1, 0.1 + GAZE_MIN_SPAN, -0.2), 0.5));
    }

    #[test]
    fn clamps_past_the_edges() {
        assert_eq!(map_gaze_axis(0.9, 0.0, -0.2, 0.2), 1.0);
        assert_eq!(map_gaze_axis(-0.9, 0.0, -0.2, 0.2), -1.0);
        // The fallback is clamped too.
        assert_eq!(map_gaze_axis(1.8, 0.0, 0.0, 0.0), 1.0);
        assert_eq!(map_gaze_axis(-1.8, 0.0, 0.0, 0.0), -1.0);
    }

    #[test]
    fn both_axes_use_their_own_edges() {
        let c = GazeCalibration { center: GazeCenter { x: 0.05, y: -0.1 }, left: 0.25, right: -0.15, up: 0.1, down: -0.3, captured_at: 0.0 };
        let (x, y) = map_gaze(0.15, -0.2, &c);
        assert!(close(x, 0.5));
        assert!(close(y, -0.5));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn json_round_trip_in_camel_case() {
        let json = r#"{ "center": { "x": 0.05, "y": -0.1 }, "left": 0.25, "right": -0.15, "up": 0.1, "down": -0.3, "capturedAt": 1700000000000 }"#;
        let c: GazeCalibration = serde_json::from_str(json).unwrap();
        assert_eq!(c, GazeCalibration { center: GazeCenter { x: 0.05, y: -0.1 }, left: 0.25, right: -0.15, up: 0.1, down: -0.3, captured_at: 1.7e12 });
        let back = serde_json::to_value(c).unwrap();
        assert_eq!(back["capturedAt"].as_f64(), Some(1.7e12));
        assert_eq!(serde_json::from_value::<GazeCalibration>(back).unwrap(), c);
    }
}
