//! Image geometry shared by the detectors and landmark models: rotated
//! regions of interest, the crop that feeds a model, SSD anchors, box
//! decoding and weighted non-maximum suppression. Coordinates are image
//! pixels (x right, y down); angles are radians, clockwise on screen, the
//! same convention as MediaPipe's `NormalizedRect.rotation`.

use crate::frame::Frame;
use std::f32::consts::PI;

/// A rotated rectangle in image pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rect {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub angle: f32,
}

impl Rect {
    /// The square that holds the whole image, centred (letterboxing).
    pub fn letterbox(width: u32, height: u32) -> Self {
        let side = width.max(height) as f32;
        Self { cx: width as f32 / 2.0, cy: height as f32 / 2.0, w: side, h: side, angle: 0.0 }
    }

    /// Maps a point in the unit square of the rect (u, v in 0..1) to image pixels.
    #[inline]
    pub fn to_image(&self, u: f32, v: f32) -> (f32, f32) {
        let (sin, cos) = self.angle.sin_cos();
        let x = (u - 0.5) * self.w;
        let y = (v - 0.5) * self.h;
        (self.cx + x * cos - y * sin, self.cy + x * sin + y * cos)
    }

    /// MediaPipe's RectTransformationCalculator: shift (in rect sizes, along
    /// the rect's own axes), make square on the long side, then scale.
    pub fn transformed(mut self, scale: f32, shift_x: f32, shift_y: f32, square_long: bool) -> Self {
        let (sin, cos) = self.angle.sin_cos();
        let dx = self.w * shift_x;
        let dy = self.h * shift_y;
        self.cx += dx * cos - dy * sin;
        self.cy += dx * sin + dy * cos;
        if square_long {
            let side = self.w.max(self.h);
            self.w = side;
            self.h = side;
        }
        self.w *= scale;
        self.h *= scale;
        self
    }

    /// Axis-aligned bounds (x0, y0, x1, y1) of the rotated rect.
    pub fn bounds(&self) -> (f32, f32, f32, f32) {
        let mut b = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for (u, v) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
            let (x, y) = self.to_image(u, v);
            b = (b.0.min(x), b.1.min(y), b.2.max(x), b.3.max(y));
        }
        b
    }

    /// Overlap of the two rects with their rotation ignored, 0-1 (MediaPipe's
    /// association compares rects this way; the rotated bounds would be larger).
    pub fn iou(&self, other: &Rect) -> f32 {
        let unrotated = |r: &Rect| (r.cx - r.w / 2.0, r.cy - r.h / 2.0, r.cx + r.w / 2.0, r.cy + r.h / 2.0);
        iou(unrotated(self), unrotated(other))
    }
}

/// Intersection over union of two (x0, y0, x1, y1) boxes.
pub fn iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let w = (a.2.min(b.2) - a.0.max(b.0)).max(0.0);
    let h = (a.3.min(b.3) - a.1.max(b.1)).max(0.0);
    let inter = w * h;
    let union = (a.2 - a.0) * (a.3 - a.1) + (b.2 - b.0) * (b.3 - b.1) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

pub fn normalize_radians(angle: f32) -> f32 {
    angle - 2.0 * PI * ((angle + PI) / (2.0 * PI)).floor()
}

/// MediaPipe's rotation from two keypoints: the angle that turns the
/// direction start→end to `target` (radians, counter-clockwise from +x with y up).
pub fn rotation_between(start: (f32, f32), end: (f32, f32), target: f32) -> f32 {
    normalize_radians(target - (-(end.1 - start.1)).atan2(end.0 - start.0))
}

/// How a crop treats pixels outside the image.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Border {
    Zero,
    Replicate,
}

/// Samples `rect` of `frame` into an NHWC float tensor of `size`x`size`
/// (bilinear), with values mapped from 0..255 to `lo..hi`. This is
/// MediaPipe's ImageToTensorCalculator: the same affine warp for the
/// letterboxed detector input and for the rotated landmark crops.
pub fn crop_to_tensor(frame: &Frame, rect: &Rect, size: usize, lo: f32, hi: f32, border: Border, out: &mut [f32]) {
    debug_assert_eq!(out.len(), size * size * 3);
    let width = frame.width as i32;
    let height = frame.height as i32;
    let stride = frame.width as usize * 3;
    let pixels = &frame.rgb[..];
    let scale = (hi - lo) / 255.0;
    let (sin, cos) = rect.angle.sin_cos();
    // Image position of output pixel (i, j) centre: origin + i * du + j * dv.
    let step_u = rect.w / size as f32;
    let step_v = rect.h / size as f32;
    let du = (cos * step_u, sin * step_u);
    let dv = (-sin * step_v, cos * step_v);
    let start = rect.to_image(0.5 / size as f32, 0.5 / size as f32);
    // Pixel centres are at +0.5: sample coordinates are shifted by that.
    let origin = (start.0 - 0.5, start.1 - 0.5);
    let fetch = |x: i32, y: i32| -> Option<usize> {
        if x >= 0 && y >= 0 && x < width && y < height {
            Some(y as usize * stride + x as usize * 3)
        } else if border == Border::Replicate {
            Some(y.clamp(0, height - 1) as usize * stride + x.clamp(0, width - 1) as usize * 3)
        } else {
            None
        }
    };
    let mut index = 0;
    for j in 0..size {
        let row_x = origin.0 + dv.0 * j as f32;
        let row_y = origin.1 + dv.1 * j as f32;
        for i in 0..size {
            let x = row_x + du.0 * i as f32;
            let y = row_y + du.1 * i as f32;
            let x0 = x.floor();
            let y0 = y.floor();
            let fx = x - x0;
            let fy = y - y0;
            let (xi, yi) = (x0 as i32, y0 as i32);
            let inside = xi >= 0 && yi >= 0 && xi + 1 < width && yi + 1 < height;
            let mut rgb = [0f32; 3];
            if inside {
                let p00 = yi as usize * stride + xi as usize * 3;
                let p10 = p00 + 3;
                let p01 = p00 + stride;
                let p11 = p01 + 3;
                let w00 = (1.0 - fx) * (1.0 - fy);
                let w10 = fx * (1.0 - fy);
                let w01 = (1.0 - fx) * fy;
                let w11 = fx * fy;
                for c in 0..3 {
                    rgb[c] = pixels[p00 + c] as f32 * w00 + pixels[p10 + c] as f32 * w10 + pixels[p01 + c] as f32 * w01 + pixels[p11 + c] as f32 * w11;
                }
            } else {
                let taps = [(xi, yi, (1.0 - fx) * (1.0 - fy)), (xi + 1, yi, fx * (1.0 - fy)), (xi, yi + 1, (1.0 - fx) * fy), (xi + 1, yi + 1, fx * fy)];
                for (tx, ty, weight) in taps {
                    if let Some(p) = fetch(tx, ty) {
                        for c in 0..3 {
                            rgb[c] += pixels[p + c] as f32 * weight;
                        }
                    }
                }
            }
            out[index] = lo + rgb[0] * scale;
            out[index + 1] = lo + rgb[1] * scale;
            out[index + 2] = lo + rgb[2] * scale;
            index += 3;
        }
    }
}

/// SSD anchor centres (MediaPipe's SsdAnchorsCalculator with
/// `fixed_anchor_size`, one aspect ratio and the interpolated scale: two
/// anchors per cell per layer, consecutive layers of one stride share a grid).
pub fn ssd_anchors(input_size: usize, strides: &[usize]) -> Vec<(f32, f32)> {
    let mut anchors = Vec::new();
    let mut layer = 0;
    while layer < strides.len() {
        let stride = strides[layer];
        let mut repeats = 0;
        while layer < strides.len() && strides[layer] == stride {
            repeats += 1;
            layer += 1;
        }
        let grid = input_size.div_ceil(stride);
        for y in 0..grid {
            for x in 0..grid {
                for _ in 0..repeats * 2 {
                    anchors.push(((x as f32 + 0.5) / grid as f32, (y as f32 + 0.5) / grid as f32));
                }
            }
        }
    }
    anchors
}

/// A detection in image pixels.
#[derive(Clone, Debug, Default)]
pub struct Detection {
    pub score: f32,
    /// x0, y0, x1, y1
    pub bounds: (f32, f32, f32, f32),
    pub keypoints: Vec<(f32, f32)>,
}

/// Decodes SSD outputs (MediaPipe TensorsToDetections: x, y, w, h then
/// keypoint pairs, offsets in input pixels from the anchor centre) into image
/// detections through the letterbox `rect`, keeping scores >= `threshold`.
#[allow(clippy::too_many_arguments)]
pub fn decode_detections(boxes: &[f32], scores: &[f32], anchors: &[(f32, f32)], values_per_box: usize, keypoints: usize, input_size: f32, threshold: f32, rect: &Rect, out: &mut Vec<Detection>) {
    out.clear();
    for (index, anchor) in anchors.iter().enumerate() {
        let score = sigmoid(scores[index].clamp(-100.0, 100.0));
        if score < threshold {
            continue;
        }
        let raw = &boxes[index * values_per_box..(index + 1) * values_per_box];
        let cx = raw[0] / input_size + anchor.0;
        let cy = raw[1] / input_size + anchor.1;
        let w = raw[2] / input_size;
        let h = raw[3] / input_size;
        let a = rect.to_image(cx - w / 2.0, cy - h / 2.0);
        let b = rect.to_image(cx + w / 2.0, cy + h / 2.0);
        let mut points = Vec::with_capacity(keypoints);
        for k in 0..keypoints {
            points.push(rect.to_image(raw[4 + k * 2] / input_size + anchor.0, raw[5 + k * 2] / input_size + anchor.1));
        }
        out.push(Detection { score, bounds: (a.0.min(b.0), a.1.min(b.1), a.0.max(b.0), a.1.max(b.1)), keypoints: points });
    }
}

/// MediaPipe's weighted non-maximum suppression: overlapping detections are
/// averaged, weighted by score, keeping the best score.
pub fn weighted_nms(mut detections: Vec<Detection>, iou_threshold: f32, max: usize) -> Vec<Detection> {
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept = Vec::new();
    while !detections.is_empty() && kept.len() < max {
        let top = detections[0].bounds;
        let (cluster, rest): (Vec<_>, Vec<_>) = detections.into_iter().partition(|d| iou(top, d.bounds) > iou_threshold);
        detections = rest;
        let total: f32 = cluster.iter().map(|d| d.score).sum();
        let mut merged = Detection { score: cluster[0].score, bounds: (0.0, 0.0, 0.0, 0.0), keypoints: vec![(0.0, 0.0); cluster[0].keypoints.len()] };
        for d in &cluster {
            let weight = d.score / total;
            merged.bounds.0 += d.bounds.0 * weight;
            merged.bounds.1 += d.bounds.1 * weight;
            merged.bounds.2 += d.bounds.2 * weight;
            merged.bounds.3 += d.bounds.3 * weight;
            for (m, k) in merged.keypoints.iter_mut().zip(&d.keypoints) {
                m.0 += k.0 * weight;
                m.1 += k.1 * weight;
            }
        }
        kept.push(merged);
    }
    kept
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The rect of a detection, rotated so keypoint `start`→`end` points along `target`.
pub fn detection_rect(detection: &Detection, start: usize, end: usize, target: f32) -> Rect {
    let b = detection.bounds;
    Rect { cx: (b.0 + b.2) / 2.0, cy: (b.1 + b.3) / 2.0, w: b.2 - b.0, h: b.3 - b.1, angle: rotation_between(detection.keypoints[start], detection.keypoints[end], target) }
}

/// The smallest rect at `angle` around `points` (HandLandmarksToRectCalculator).
pub fn rotated_bounds(points: impl Iterator<Item = (f32, f32)> + Clone, angle: f32) -> Rect {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for (x, y) in points.clone() {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let (acx, acy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let (sin_r, cos_r) = (-angle).sin_cos();
    let (mut px0, mut py0, mut px1, mut py1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for (x, y) in points {
        let (ox, oy) = (x - acx, y - acy);
        let px = ox * cos_r - oy * sin_r;
        let py = ox * sin_r + oy * cos_r;
        px0 = px0.min(px);
        py0 = py0.min(py);
        px1 = px1.max(px);
        py1 = py1.max(py);
    }
    let (pcx, pcy) = ((px0 + px1) / 2.0, (py0 + py1) / 2.0);
    let (sin, cos) = angle.sin_cos();
    Rect { cx: pcx * cos - pcy * sin + acx, cy: pcx * sin + pcy * cos + acy, w: px1 - px0, h: py1 - py0, angle }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_counts_match_models() {
        assert_eq!(ssd_anchors(128, &[8, 16, 16, 16]).len(), 896);
        assert_eq!(ssd_anchors(192, &[8, 16, 16, 16]).len(), 2016);
    }

    #[test]
    fn rect_round_trip_and_rotation() {
        let rect = Rect { cx: 100.0, cy: 50.0, w: 40.0, h: 20.0, angle: PI / 2.0 };
        let (x, y) = rect.to_image(1.0, 0.5);
        assert!((x - 100.0).abs() < 1e-4 && (y - 70.0).abs() < 1e-4, "{x} {y}");
        // Wrist below the knuckles (y down): already upright, no rotation needed for target 90 degrees.
        assert!(rotation_between((0.0, 10.0), (0.0, 0.0), PI / 2.0).abs() < 1e-6);
    }

    #[test]
    fn letterbox_crop_samples_image() {
        let mut frame = Frame::new(0, 4, 2, vec![0; 24]);
        for p in frame.rgb.chunks_mut(3) {
            p.copy_from_slice(&[255, 0, 0]);
        }
        let mut out = vec![0.0; 4 * 4 * 3];
        crop_to_tensor(&frame, &Rect::letterbox(4, 2), 4, 0.0, 1.0, Border::Zero, &mut out);
        // Middle rows are image, top and bottom rows are padding.
        assert!(out[(4 + 1) * 3] > 0.9);
        assert!(out[3] < 0.6);
    }

    #[test]
    fn nms_merges_overlaps() {
        let a = Detection { score: 0.9, bounds: (0.0, 0.0, 10.0, 10.0), keypoints: vec![(1.0, 1.0)] };
        let b = Detection { score: 0.6, bounds: (1.0, 1.0, 11.0, 11.0), keypoints: vec![(2.0, 2.0)] };
        let c = Detection { score: 0.7, bounds: (50.0, 50.0, 60.0, 60.0), keypoints: vec![(55.0, 55.0)] };
        let kept = weighted_nms(vec![a, b, c], 0.3, 5);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].bounds.0 - 0.4).abs() < 1e-4);
    }
}
