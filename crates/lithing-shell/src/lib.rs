//! Phase 3 skeleton: the native UI shell (egui / Slint / iced, see
//! ARCHITECTURE.md) and, the hard part, a host that keeps today's
//! JavaScript extensions (the app's SDK: api.network, api.hotkeys, UI
//! panels...) running without a web view.
//!
//! The plan is an [`ExtensionRuntime`] trait with two implementations:
//! an embedded JS engine (rquickjs: small, no JIT, easy to sandbox; or
//! deno_core/V8 for speed and modern JS) exposing the same `api.*` surface as
//! src/app/extension-runtime.ts, and optionally WASM components (wasmtime)
//! for new extensions. UI that extensions declare (sdk-ui.ts) is described as
//! data, so the native shell can render it with its own widgets.

use std::collections::BTreeMap;

/// A host call an extension makes (`api.<namespace>.<method>(args)`).
#[derive(Clone, Debug, PartialEq)]
pub struct HostCall {
    pub namespace: String,
    pub method: String,
    pub args: serde_json::Value,
}

/// Runs extensions. Implementations: QuickJS (rquickjs) first, WASM later.
pub trait ExtensionRuntime {
    /// Loads an extension's entry script with its manifest's permissions.
    fn load(&mut self, id: &str, source: &str, permissions: &[String]) -> Result<(), String>;
    /// Delivers an event (tracking frame summary, hotkey, network reply...).
    fn dispatch(&mut self, id: &str, event: &str, payload: &serde_json::Value) -> Result<(), String>;
    /// Host calls the extensions made since the last poll, to be served by the shell.
    fn drain_calls(&mut self) -> Vec<(String, HostCall)>;
}

/// UI an extension declares (the sdk-ui.ts model as data), rendered by the native shell.
#[derive(Clone, Debug, PartialEq)]
pub enum Widget {
    Label(String),
    Button { id: String, label: String },
    Slider { id: String, label: String, min: f64, max: f64, value: f64 },
    Toggle { id: String, label: String, value: bool },
    Group { title: String, children: Vec<Widget> },
}

/// A permission check shared by every runtime (the manifest's grants).
pub fn allowed(permissions: &[String], call: &HostCall) -> bool {
    let wanted = format!("{}.{}", call.namespace, call.method);
    permissions.iter().any(|p| p == "*" || *p == call.namespace || *p == wanted)
}

/// Settings panels by extension id.
pub type Panels = BTreeMap<String, Vec<Widget>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permissions_by_namespace_or_method() {
        let call = HostCall { namespace: "network".into(), method: "fetch".into(), args: serde_json::Value::Null };
        assert!(allowed(&["network".into()], &call));
        assert!(allowed(&["network.fetch".into()], &call));
        assert!(!allowed(&["hotkeys".into()], &call));
    }
}
