//! Thread priorities: tracking yields to a game but keeps up with the camera.

/// Priority of the capture and inference threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub enum ThreadPriority {
    Lowest,
    /// The default for inference: a game at normal priority goes first.
    #[default]
    BelowNormal,
    Normal,
    /// For capture: a camera frame should never wait behind anything else of ours.
    AboveNormal,
}

impl ThreadPriority {
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "lowest" | "idle" => Some(Self::Lowest),
            "belownormal" | "low" => Some(Self::BelowNormal),
            "normal" => Some(Self::Normal),
            "abovenormal" | "high" => Some(Self::AboveNormal),
            _ => None,
        }
    }
}

/// Sets the calling thread's priority (best effort; failures are ignored).
pub fn set_current(priority: ThreadPriority) {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL, THREAD_PRIORITY_BELOW_NORMAL, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_NORMAL};
        let value = match priority {
            ThreadPriority::Lowest => THREAD_PRIORITY_LOWEST,
            ThreadPriority::BelowNormal => THREAD_PRIORITY_BELOW_NORMAL,
            ThreadPriority::Normal => THREAD_PRIORITY_NORMAL,
            ThreadPriority::AboveNormal => THREAD_PRIORITY_ABOVE_NORMAL,
        };
        SetThreadPriority(GetCurrentThread(), value);
    }
    #[cfg(unix)]
    unsafe {
        // Per-thread nice value on Linux (raising priority needs CAP_SYS_NICE; ignored otherwise).
        let nice = match priority {
            ThreadPriority::Lowest => 15,
            ThreadPriority::BelowNormal => 5,
            ThreadPriority::Normal => 0,
            ThreadPriority::AboveNormal => -5,
        };
        #[cfg(target_os = "linux")]
        {
            let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
            libc::setpriority(libc::PRIO_PROCESS, tid, nice);
        }
        #[cfg(not(target_os = "linux"))]
        let _ = nice;
    }
}
