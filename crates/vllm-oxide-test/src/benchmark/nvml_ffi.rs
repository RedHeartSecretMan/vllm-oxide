//! Linux NVML ABI, selected by its standard soname and bound to its real loaded file.
//! No object or opaque device handle crosses threads: the worker opens and closes it.

use anyhow::{bail, Result};
use serde::Serialize;

#[derive(Clone, Copy, Default, Serialize)]
#[repr(C)]
pub(super) struct MemoryBytes {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}

pub(super) struct Reply {
    pub code: i32,
    pub memory: Option<MemoryBytes>,
    pub error: Option<String>,
}

pub(super) trait SamplingApi {
    fn metadata(&self) -> serde_json::Value;
    fn initialize(&mut self) -> Result<()>;
    fn query(&mut self) -> Reply;
    fn close(&mut self) -> Result<()>;
}

pub(super) fn open() -> Result<Box<dyn SamplingApi>> {
    #[cfg(target_os = "linux")]
    return Ok(Box::new(linux::Session::open()?));
    #[cfg(not(target_os = "linux"))]
    bail!("the direct NVML memory method requires Linux")
}

// This is the only unsafe boundary: Linux dlopen/dlsym and the documented NVML C ABI.
// Every function pointer remains valid until shutdown and dlclose on its owning worker.
#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
mod linux {
    use super::*;
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::path::PathBuf;
    use std::time::Instant;

    use anyhow::Context;
    use sha2::{Digest, Sha256};

    const LIBRARY: &str = "libnvidia-ml.so.1";
    type Init = unsafe extern "C" fn() -> c_int;
    type Handle = unsafe extern "C" fn(u32, *mut *mut c_void) -> c_int;
    type Query = unsafe extern "C" fn(*mut c_void, *mut MemoryBytes) -> c_int;
    type Version = unsafe extern "C" fn(*mut c_char, u32) -> c_int;
    type ErrorString = unsafe extern "C" fn(c_int) -> *const c_char;

    #[repr(C)]
    struct LinkMap {
        address: usize,
        name: *const c_char,
        dynamic: *mut c_void,
        next: *mut LinkMap,
        previous: *mut LinkMap,
    }

    #[repr(C)]
    struct AddressInfo {
        filename: *const c_char,
        base: *mut c_void,
        symbol_name: *const c_char,
        symbol_address: *mut c_void,
    }

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlerror() -> *const c_char;
        fn dlinfo(handle: *mut c_void, request: c_int, result: *mut c_void) -> c_int;
        fn dladdr(address: *const c_void, result: *mut AddressInfo) -> c_int;
    }

    fn loader_error() -> String {
        // SAFETY: dlerror returns NULL or a thread-local, NUL-terminated loader string.
        let pointer = unsafe { dlerror() };
        if pointer.is_null() {
            "dynamic loader returned no error text".into()
        } else {
            // SAFETY: non-NULL dlerror strings stay valid until the next loader call.
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        }
    }

    struct Library(*mut c_void);

    impl Library {
        fn open() -> Result<Self> {
            // No application environment variable or CLI argument can replace this soname.
            Self::open_name(c"libnvidia-ml.so.1")
        }

        fn open_name(name: &CStr) -> Result<Self> {
            // SAFETY: name is NUL-terminated; RTLD_NOW|RTLD_LOCAL resolves before use.
            let pointer = unsafe { dlopen(name.as_ptr(), 2) };
            if pointer.is_null() {
                bail!(
                    "loading NVML soname {}: {}",
                    name.to_string_lossy(),
                    loader_error()
                );
            }
            Ok(Self(pointer))
        }

        fn loaded_path(&self) -> Result<PathBuf> {
            let mut map: *mut LinkMap = std::ptr::null_mut();
            // SAFETY: live dlopen handle; RTLD_DI_LINKMAP writes one LinkMap pointer slot.
            if unsafe { dlinfo(self.0, 2, (&raw mut map).cast()) } != 0 || map.is_null() {
                bail!("resolving actual loaded NVML object: {}", loader_error());
            }
            // SAFETY: the dynamic loader owns this link map for the lifetime of self.
            let name = unsafe { (*map).name };
            path_from_loader(name)
        }

        fn symbol<T: Copy>(&self, name: &[u8]) -> Result<T> {
            if std::mem::size_of::<T>() != std::mem::size_of::<*mut c_void>() {
                bail!("invalid NVML function pointer size");
            }
            // SAFETY: private callers pass constant NUL-terminated names and a live library.
            let pointer = unsafe { dlsym(self.0, name.as_ptr().cast()) };
            if pointer.is_null() {
                bail!(
                    "NVML symbol {}: {}",
                    String::from_utf8_lossy(name),
                    loader_error()
                );
            }
            // SAFETY: private callers name the exact documented extern-C function type;
            // Linux represents these callable dlsym addresses as pointer-sized values.
            Ok(unsafe { std::mem::transmute_copy(&pointer) })
        }

        fn close(&mut self) -> Result<()> {
            if !self.0.is_null() {
                let pointer = std::mem::replace(&mut self.0, std::ptr::null_mut());
                // SAFETY: this handle is owned, and no symbol will be called after close.
                if unsafe { dlclose(pointer) } != 0 {
                    bail!("NVML dlclose: {}", loader_error());
                }
            }
            Ok(())
        }
    }

    fn path_from_loader(name: *const c_char) -> Result<PathBuf> {
        if name.is_null() {
            bail!("dynamic loader omitted the loaded object filename");
        }
        // SAFETY: caller passes a loader-owned NUL-terminated name while its library is live.
        let path = PathBuf::from(unsafe { CStr::from_ptr(name) }.to_str()?);
        if !path.is_absolute() {
            bail!("dynamic loader did not provide an absolute object path");
        }
        Ok(path)
    }

    fn symbol_file(address: *const c_void) -> Result<serde_json::Value> {
        let mut info = AddressInfo {
            filename: std::ptr::null(),
            base: std::ptr::null_mut(),
            symbol_name: std::ptr::null(),
            symbol_address: std::ptr::null_mut(),
        };
        // SAFETY: address is a resolved live function; info is the documented four-pointer ABI.
        if unsafe { dladdr(address, &mut info) } == 0 {
            bail!("dynamic loader cannot identify NVML memory function");
        }
        let path = path_from_loader(info.filename)?;
        file_identity(&path)
    }

    fn file_identity(path: &std::path::Path) -> Result<serde_json::Value> {
        let canonical = std::fs::canonicalize(path)?;
        let bytes = std::fs::read(&canonical)?;
        Ok(
            serde_json::json!({"path":path,"canonical_path":canonical,"sha256":format!("{:x}",Sha256::digest(&bytes)),"size_bytes":bytes.len()}),
        )
    }

    impl Drop for Library {
        fn drop(&mut self) {
            if let Err(error) = self.close() {
                eprintln!("NVML_MEMORY_LOADER_CLEANUP_FAILURE {error:#}");
            }
        }
    }

    pub(super) struct Session {
        library: Library,
        init: Init,
        handle_fn: Handle,
        query_fn: Query,
        shutdown: Init,
        error_fn: ErrorString,
        nvml_version: Version,
        driver_version: Version,
        device: *mut c_void,
        initialized: bool,
        closed: bool,
        metadata: serde_json::Value,
    }

    impl Session {
        pub(super) fn open() -> Result<Self> {
            let library = Library::open()?;
            let loaded = library.loaded_path()?;
            let identity = file_identity(&loaded)?;
            let query_fn: Query = library.symbol(b"nvmlDeviceGetMemoryInfo\0")?;
            Ok(Self {
                init: library.symbol(b"nvmlInit_v2\0")?,
                handle_fn: library.symbol(b"nvmlDeviceGetHandleByIndex_v2\0")?,
                query_fn,
                shutdown: library.symbol(b"nvmlShutdown\0")?,
                error_fn: library.symbol(b"nvmlErrorString\0")?,
                nvml_version: library.symbol(b"nvmlSystemGetNVMLVersion\0")?,
                driver_version: library.symbol(b"nvmlSystemGetDriverVersion\0")?,
                metadata: serde_json::json!({
                    "requested_soname":LIBRARY,
                    "loader_selection":"fixed soname through platform dynamic linker; no application library override",
                    "loaded_library_path":identity["path"],
                    "canonical_library_path":identity["canonical_path"],
                    "library_sha256":identity["sha256"],
                    "library_size_bytes":identity["size_bytes"],
                    "memory_query_symbol_file":symbol_file(query_fn as *const c_void)?,
                    "actual_library_mappings":std::fs::read_to_string("/proc/self/maps")?.lines()
                        .filter(|line| line.contains("libnvidia-ml.so")).collect::<Vec<_>>(),
                    "device_index":0,"memory_size":std::mem::size_of::<MemoryBytes>(),
                    "memory_offsets":[std::mem::offset_of!(MemoryBytes,total),
                        std::mem::offset_of!(MemoryBytes,free),std::mem::offset_of!(MemoryBytes,used)],
                }),
                library,
                device: std::ptr::null_mut(),
                initialized: false,
                closed: false,
            })
        }

        fn error(&self, code: i32) -> String {
            // SAFETY: a loaded error function accepts every NVML return enum value.
            let text = unsafe { (self.error_fn)(code) };
            if text.is_null() {
                return format!("NVML code {code} with NULL error string");
            }
            // SAFETY: NVML returns a library-owned, NUL-terminated constant string.
            format!(
                "NVML code {code}: {}",
                unsafe { CStr::from_ptr(text) }.to_string_lossy()
            )
        }

        fn version(&self, function: Version) -> Result<String> {
            let mut buffer = [0u8; 96];
            // SAFETY: buffer is writable for 96 bytes; the explicit length matches it.
            let code = unsafe { function(buffer.as_mut_ptr().cast(), 96) };
            if code != 0 {
                bail!("{}", self.error(code));
            }
            let nul = buffer
                .iter()
                .position(|byte| *byte == 0)
                .context("unterminated NVML version")?;
            Ok(std::str::from_utf8(&buffer[..nul])?.to_owned())
        }
    }

    impl SamplingApi for Session {
        fn metadata(&self) -> serde_json::Value {
            self.metadata.clone()
        }

        fn initialize(&mut self) -> Result<()> {
            let started = Instant::now();
            // SAFETY: this is the resolved zero-argument initializer; session owns its ref.
            let code = unsafe { (self.init)() };
            self.metadata["init_wall_span_ns"] = serde_json::json!(started.elapsed().as_nanos());
            if code != 0 {
                bail!("nvmlInit_v2: {}", self.error(code));
            }
            self.initialized = true;
            // SAFETY: output is a live pointer slot; index0 is the explicitly selected device.
            let code = unsafe { (self.handle_fn)(0, &mut self.device) };
            if code != 0 {
                bail!("nvmlDeviceGetHandleByIndex_v2: {}", self.error(code));
            }
            if self.device.is_null() {
                bail!("NVML returned a NULL device handle");
            }
            self.metadata["nvml_version"] = self.version(self.nvml_version)?.into();
            self.metadata["driver_version"] = self.version(self.driver_version)?.into();
            Ok(())
        }

        fn query(&mut self) -> Reply {
            let mut memory = MemoryBytes::default();
            // SAFETY: initialized opaque handle and writable repr-C 24-byte v1 memory.
            let code = unsafe { (self.query_fn)(self.device, &mut memory) };
            Reply {
                code,
                memory: (code == 0).then_some(memory),
                error: (code != 0).then(|| self.error(code)),
            }
        }

        fn close(&mut self) -> Result<()> {
            self.closed = true;
            let mut errors = Vec::new();
            if self.initialized {
                self.initialized = false;
                // SAFETY: exactly one shutdown attempt for this session's successful init.
                let code = unsafe { (self.shutdown)() };
                if code != 0 {
                    errors.push(format!("nvmlShutdown: {}", self.error(code)));
                }
            }
            if let Err(error) = self.library.close() {
                errors.push(format!("{error:#}"));
            }
            if !errors.is_empty() {
                bail!("{}", errors.join("; "));
            }
            Ok(())
        }
    }

    impl Drop for Session {
        fn drop(&mut self) {
            if !self.closed {
                if let Err(error) = self.close() {
                    eprintln!("NVML_MEMORY_SESSION_CLEANUP_FAILURE {error:#}");
                }
            }
        }
    }

    #[cfg(test)]
    #[allow(clippy::unwrap_used)]
    mod tests {
        use super::*;

        #[test]
        fn nvml_loader_policy_reports_standard_soname_and_actual_loaded_file() {
            // This opens the DSO but never calls NVML init/query or creates a GPU handle.
            match Session::open() {
                Ok(mut session) => {
                    let info = session.metadata();
                    session.close().unwrap();
                    assert_eq!(info["requested_soname"], "libnvidia-ml.so.1");
                    let loaded = info["loaded_library_path"].as_str().unwrap();
                    assert!(std::path::Path::new(loaded).is_absolute());
                    assert_eq!(
                        info["library_sha256"],
                        format!("{:x}", Sha256::digest(std::fs::read(loaded).unwrap()))
                    );
                }
                Err(error) => {
                    // CPU CI without the vendor library must fail explicitly, never fabricate it.
                    let message = format!("{error:#}");
                    assert!(
                        message.contains("loading NVML soname libnvidia-ml.so.1")
                            || message.contains("NVML symbol"),
                        "{message}"
                    );
                    assert!(!message.contains("/usr/lib/wsl"));
                }
            }
        }

        #[test]
        fn nvml_loader_missing_library_and_required_symbol_fail_explicitly() {
            assert!(Library::open_name(c"vllm-oxide-missing-nvml-cpu-fixture.so.1").is_err());
            let mut library = Library::open_name(c"libc.so.6").unwrap();
            let path = library.loaded_path().unwrap();
            assert!(path.is_absolute());
            assert!(library.symbol::<Init>(b"nvmlInit_v2\0").is_err());
            library.close().unwrap();
        }
    }
}
