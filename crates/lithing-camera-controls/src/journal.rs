//! Crash safety for control changes: the original values are written to a
//! small file before the first change and removed once they are restored. If
//! the process dies in between (killed, crashed, power loss), the next open
//! of the same device finds the file and puts the originals back first, so a
//! camera that remembers its settings across apps is never left changed.

use crate::{ControlKind, ControlValue};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Journal {
    path: PathBuf,
}

impl Journal {
    /// The journal of `device_id` in `dir` (default: the temp directory).
    pub fn for_device(dir: Option<&Path>, device_id: &str) -> Self {
        let dir = dir.map(Path::to_path_buf).unwrap_or_else(std::env::temp_dir);
        Self { path: dir.join(format!("lithing-camera-restore-{:016x}.txt", fnv1a(device_id.to_ascii_lowercase().as_bytes()))) }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Writes the originals (first line: the device, then `kind value auto`).
    pub fn save(&self, device_id: &str, originals: &[(ControlKind, ControlValue)]) -> std::io::Result<()> {
        let mut text = format!("{device_id}\n");
        for (kind, value) in originals {
            text.push_str(&format!("{} {} {}\n", kind.key(), value.value, u8::from(value.auto)));
        }
        let temp = self.path.with_extension("tmp");
        std::fs::write(&temp, text)?;
        std::fs::rename(&temp, &self.path)
    }

    /// Originals left by a run that did not restore them.
    pub fn load(&self) -> Option<Vec<(ControlKind, ControlValue)>> {
        let text = std::fs::read_to_string(&self.path).ok()?;
        let entries: Vec<_> = text
            .lines()
            .skip(1)
            .filter_map(|line| {
                let mut parts = line.split_whitespace();
                let kind = ControlKind::from_key(parts.next()?)?;
                let value = parts.next()?.parse().ok()?;
                let auto = parts.next()? == "1";
                Some((kind, ControlValue { value, auto }))
            })
            .collect();
        (!entries.is_empty()).then_some(entries)
    }

    pub fn clear(&self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, &b| (hash ^ b as u64).wrapping_mul(0x100000001b3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_load_clear() {
        let dir = std::env::temp_dir().join(format!("lithing-journal-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let journal = Journal::for_device(Some(&dir), r"\\?\usb#vid_0c45&pid_636b");
        assert!(journal.load().is_none());
        let originals = [(ControlKind::AutoExposurePriority, ControlValue::manual(1)), (ControlKind::Exposure, ControlValue { value: -4, auto: true })];
        journal.save("dev", &originals).unwrap();
        assert_eq!(journal.load().unwrap(), originals.to_vec());
        // The same device under another case maps to the same file.
        assert_eq!(Journal::for_device(Some(&dir), r"\\?\USB#VID_0C45&PID_636B").path(), journal.path());
        journal.clear();
        assert!(journal.load().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
