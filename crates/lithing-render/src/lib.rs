//! Phase 2 skeleton: the native renderer that would replace three.js +
//! @pixiv/three-vrm. Nothing here draws yet; it pins down the boundaries so
//! the work can be split (see ARCHITECTURE.md, "Phase 2"):
//!
//! * [`AvatarInput`]: what the renderer consumes each frame, built from a
//!   [`lithing_tracking::TrackingFrame`] (the same values the web
//!   AvatarController reads today).
//! * [`Avatar`]: a loaded VRM 0.x/1.0 model (glTF + VRMC_vrm / VRM extension):
//!   humanoid bones, expressions, look-at, spring bones, MToon materials.
//! * [`Renderer`]: the wgpu backend, rendering to a window surface, to an
//!   offscreen texture for the virtual camera, or both.

use lithing_tracking::{HandPose, TrackingData, TrackingFrame};

/// Per-frame input to the avatar, independent of where it came from
/// (native tracker, phone link, network).
#[derive(Clone, Debug, Default)]
pub struct AvatarInput {
    pub tracking: TrackingData,
    /// Blendshapes by avatar-side name (52 ARKit-style + tongue channels).
    pub blendshapes: Vec<(&'static str, f32)>,
    pub hands: Vec<HandPose>,
    /// Seconds since the previous input, for springs and smoothing.
    pub delta: f32,
}

impl AvatarInput {
    pub fn from_frame(frame: &TrackingFrame, delta: f32) -> Self {
        let blendshapes = frame
            .blendshapes
            .map(|values| values.iter().enumerate().map(|(i, v)| (lithing_tracking::channel_name(i), *v)).collect())
            .unwrap_or_default();
        Self { tracking: frame.tracking.unwrap_or_default(), blendshapes, hands: frame.hands.clone(), delta }
    }
}

/// VRM spec version of a loaded model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VrmVersion {
    /// `VRM` glTF extension (UniVRM 0.x).
    V0,
    /// `VRMC_vrm` 1.0 (+ VRMC_springBone, VRMC_materials_mtoon).
    V1,
}

/// A loaded avatar (phase 2 work item: glTF via the `gltf` crate + VRM extensions).
pub trait Avatar {
    fn version(&self) -> VrmVersion;
    /// Expression names the model defines (presets and custom).
    fn expressions(&self) -> Vec<String>;
    /// Applies one frame: head/neck/spine from TrackingData, arms and fingers
    /// from HandPose (IK as in the web AvatarController), expressions through
    /// the user's mapping, then spring bones for `input.delta`.
    fn apply(&mut self, input: &AvatarInput);
}

/// Where a frame goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Window,
    /// Offscreen RGBA texture read back for the virtual camera / OBS.
    Offscreen { width: u32, height: u32 },
}

/// The renderer backend (phase 2 work item: wgpu, MToon port, outlines).
pub trait Renderer {
    type Avatar: Avatar;
    fn render(&mut self, avatar: &Self::Avatar, target: Target);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_names_blendshapes() {
        let mut values = [0.0f32; lithing_tracking::BLENDSHAPE_CHANNELS];
        values[25] = 0.5;
        let frame = TrackingFrame { blendshapes: Some(values), ..Default::default() };
        let input = AvatarInput::from_frame(&frame, 1.0 / 60.0);
        assert_eq!(input.blendshapes[25], ("jawOpen", 0.5));
    }
}
