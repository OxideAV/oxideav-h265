//! Runtime libde265 / libx265 interop for the cross-decode fuzz harnesses.
//!
//! Both shared libraries are loaded via `dlopen` at first call — there is
//! no `de265-sys` / `x265-sys`-style build-script dep that would pull
//! libde265 or libx265 source into the workspace's cargo dep tree. Each
//! harness checks the relevant `available()` up front and `return`s
//! early when the shared library isn't installed, so fuzz binaries
//! built on a host without the libraries simply do nothing instead of
//! panicking.
//!
//! Workspace policy bars consulting libde265 / libx265 / libavcodec
//! source; we only inspect the public C headers (`<libde265/de265.h>`,
//! `<x265.h>`) for function signatures + struct layouts.
//!
//! Install on macOS with `brew install libde265 x265`. On Debian /
//! Ubuntu install `libde265-dev libx265-dev`. The loader probes the
//! conventional shared-object names for both platforms.
//!
//! ## libx265 picture struct layout
//!
//! `x265_picture` is huge and version-fragile (it embeds analysis
//! buffers, SEI metadata, Dolby Vision RPU, etc.). Rather than mirror
//! the whole struct, we ask libx265 to allocate + initialise it via the
//! `x265_picture_alloc` + `x265_picture_init` helpers, then write only
//! the `planes`, `stride`, `bitDepth`, `sliceType`, `colorSpace`,
//! `width`, and `height` fields by **byte offset** into the prefix —
//! that prefix has been stable since x265 1.x and the offsets are
//! easy to audit against the public `x265.h` header. The opaque tail
//! is left untouched (`x265_picture_init` zeroed it).

#![allow(unsafe_code)]

pub mod libde265 {
    use libloading::{Library, Symbol};
    use std::ffi::c_void;
    use std::sync::OnceLock;

    /// Conventional libde265 shared-object names the loader will try
    /// in order. Covers macOS (`.dylib`), Linux (versioned + plain
    /// `.so`), and Windows (`.dll`).
    const CANDIDATES: &[&str] = &[
        "libde265.dylib",
        "libde265.0.dylib",
        "libde265.so.0",
        "libde265.so",
        "de265.dll",
    ];

    fn lib() -> Option<&'static Library> {
        static LIB: OnceLock<Option<Library>> = OnceLock::new();
        LIB.get_or_init(|| {
            for name in CANDIDATES {
                // SAFETY: `Library::new` is documented as unsafe because
                // the loaded library may run code at load time. We
                // accept that risk for fuzz tooling — libde265 is a
                // well-behaved shared library.
                if let Ok(l) = unsafe { Library::new(name) } {
                    return Some(l);
                }
            }
            None
        })
        .as_ref()
    }

    /// True iff a libde265 shared library was successfully loaded.
    /// Cross-decode fuzz harnesses early-return when this is false so
    /// the binary still runs without an oracle (the assertions just
    /// don't fire).
    pub fn available() -> bool {
        lib().is_some()
    }

    /// An 8-bit 4:2:0 frame as decoded by libde265.
    pub struct DecodedYuv420 {
        pub width: u32,
        pub height: u32,
        /// Tightly packed Y plane (length `width * height`).
        pub y: Vec<u8>,
        /// Tightly packed Cb plane (length `(width/2) * (height/2)`).
        pub cb: Vec<u8>,
        /// Tightly packed Cr plane (length `(width/2) * (height/2)`).
        pub cr: Vec<u8>,
    }

    /// Decode an Annex-B HEVC byte stream to 8-bit YUV 4:2:0.
    /// Returns `None` if libde265 isn't available, the stream is not
    /// 4:2:0 / 8-bit, or no picture is produced.
    pub fn decode_to_yuv(data: &[u8]) -> Option<DecodedYuv420> {
        type NewDecoderFn = unsafe extern "C" fn() -> *mut c_void;
        type FreeDecoderFn = unsafe extern "C" fn(*mut c_void) -> i32;
        type PushDataFn =
            unsafe extern "C" fn(*mut c_void, *const u8, i32, i64, *mut c_void) -> i32;
        type FlushFn = unsafe extern "C" fn(*mut c_void) -> i32;
        type DecodeFn = unsafe extern "C" fn(*mut c_void, *mut i32) -> i32;
        type GetNextPicFn = unsafe extern "C" fn(*mut c_void) -> *const c_void;
        type ReleaseNextPicFn = unsafe extern "C" fn(*mut c_void);
        type ImageWidthFn = unsafe extern "C" fn(*const c_void, i32) -> i32;
        type ImageHeightFn = unsafe extern "C" fn(*const c_void, i32) -> i32;
        type ImageChromaFn = unsafe extern "C" fn(*const c_void) -> i32;
        type ImageBppFn = unsafe extern "C" fn(*const c_void, i32) -> i32;
        type ImagePlaneFn = unsafe extern "C" fn(*const c_void, i32, *mut i32) -> *const u8;
        let l = lib()?;
        unsafe {
            let new_decoder: Symbol<NewDecoderFn> = l.get(b"de265_new_decoder").ok()?;
            let free_decoder: Symbol<FreeDecoderFn> = l.get(b"de265_free_decoder").ok()?;
            let push_data: Symbol<PushDataFn> = l.get(b"de265_push_data").ok()?;
            let flush_data: Symbol<FlushFn> = l.get(b"de265_flush_data").ok()?;
            let decode: Symbol<DecodeFn> = l.get(b"de265_decode").ok()?;
            let get_next_pic: Symbol<GetNextPicFn> = l.get(b"de265_get_next_picture").ok()?;
            let release_next_pic: Symbol<ReleaseNextPicFn> =
                l.get(b"de265_release_next_picture").ok()?;
            let image_width: Symbol<ImageWidthFn> = l.get(b"de265_get_image_width").ok()?;
            let image_height: Symbol<ImageHeightFn> = l.get(b"de265_get_image_height").ok()?;
            let image_chroma: Symbol<ImageChromaFn> = l.get(b"de265_get_chroma_format").ok()?;
            let image_bpp: Symbol<ImageBppFn> = l.get(b"de265_get_bits_per_pixel").ok()?;
            let image_plane: Symbol<ImagePlaneFn> = l.get(b"de265_get_image_plane").ok()?;

            let dec = new_decoder();
            if dec.is_null() {
                return None;
            }
            // Push the entire bitstream + flush + drain decode loop.
            // libde265 returns DE265_OK == 0; non-zero may still be a
            // recoverable warning, but for the fuzz path we surface any
            // non-OK push as a None so the harness skips the assertion.
            let push_rc = push_data(
                dec,
                data.as_ptr(),
                data.len() as i32,
                0,
                std::ptr::null_mut(),
            );
            if push_rc != 0 && push_rc < 1000 {
                free_decoder(dec);
                return None;
            }
            let _ = flush_data(dec);
            // Drain the decode loop. Cap iterations defensively in case
            // of a pathological bitstream.
            let mut more: i32 = 1;
            for _ in 0..1024 {
                if more == 0 {
                    break;
                }
                let rc = decode(dec, &mut more);
                if rc != 0 && rc < 1000 {
                    // Hard error (warnings are >= 1000 in de265_error).
                    break;
                }
            }
            // Pull the first available picture.
            let pic = get_next_pic(dec);
            if pic.is_null() {
                free_decoder(dec);
                return None;
            }
            // 4:2:0 == de265_chroma_420 == 1.
            let chroma = image_chroma(pic);
            let bpp = image_bpp(pic, 0);
            if chroma != 1 || bpp != 8 {
                release_next_pic(dec);
                free_decoder(dec);
                return None;
            }
            let w = image_width(pic, 0).max(0) as usize;
            let h = image_height(pic, 0).max(0) as usize;
            let cw = image_width(pic, 1).max(0) as usize;
            let ch = image_height(pic, 1).max(0) as usize;
            if w == 0 || h == 0 || cw == 0 || ch == 0 {
                release_next_pic(dec);
                free_decoder(dec);
                return None;
            }
            let mut y_stride: i32 = 0;
            let mut cb_stride: i32 = 0;
            let mut cr_stride: i32 = 0;
            let y_ptr = image_plane(pic, 0, &mut y_stride);
            let cb_ptr = image_plane(pic, 1, &mut cb_stride);
            let cr_ptr = image_plane(pic, 2, &mut cr_stride);
            if y_ptr.is_null() || cb_ptr.is_null() || cr_ptr.is_null() {
                release_next_pic(dec);
                free_decoder(dec);
                return None;
            }
            let mut y = vec![0u8; w * h];
            for row in 0..h {
                let src = y_ptr.add(row * y_stride as usize);
                std::ptr::copy_nonoverlapping(src, y.as_mut_ptr().add(row * w), w);
            }
            let mut cb = vec![0u8; cw * ch];
            let mut cr = vec![0u8; cw * ch];
            for row in 0..ch {
                let src_cb = cb_ptr.add(row * cb_stride as usize);
                let src_cr = cr_ptr.add(row * cr_stride as usize);
                std::ptr::copy_nonoverlapping(src_cb, cb.as_mut_ptr().add(row * cw), cw);
                std::ptr::copy_nonoverlapping(src_cr, cr.as_mut_ptr().add(row * cw), cw);
            }
            release_next_pic(dec);
            free_decoder(dec);
            Some(DecodedYuv420 {
                width: w as u32,
                height: h as u32,
                y,
                cb,
                cr,
            })
        }
    }
}

pub mod libx265 {
    use libloading::{Library, Symbol};
    use std::ffi::{c_void, CString};
    use std::sync::OnceLock;

    /// Conventional libx265 shared-object names the loader will try in
    /// order. The number suffix (`215`, `216`, …) is the X265_BUILD api
    /// version and changes every time libx265 ships an ABI bump; we
    /// list a couple of the recent versions and fall back to the
    /// unversioned `.so` / `.dylib` symlink.
    const CANDIDATES: &[&str] = &[
        "libx265.dylib",
        "libx265.216.dylib",
        "libx265.215.dylib",
        "libx265.214.dylib",
        "libx265.so.216",
        "libx265.so.215",
        "libx265.so.214",
        "libx265.so",
        "x265.dll",
    ];

    /// Try the symbol both versioned (`x265_encoder_open_216`) and
    /// unversioned (`x265_encoder_open`). The header `#define`s the
    /// short name to the versioned one for ABI safety, but some
    /// distributions also export the short name.
    fn lookup<'lib, T>(l: &'lib Library, base: &str) -> Option<Symbol<'lib, T>> {
        // SAFETY: `Library::get` is unsafe because the looked-up symbol
        // could be of the wrong type. The caller assigns the returned
        // `Symbol` to a `Symbol<unsafe extern "C" fn(...)>` whose
        // signature matches the documented x265 prototype.
        unsafe {
            for build in [
                216u32, 215, 214, 213, 212, 211, 210, 209, 208, 207, 206, 205,
            ] {
                let versioned = format!("{base}_{build}");
                if let Ok(s) = l.get::<T>(versioned.as_bytes()) {
                    return Some(s);
                }
            }
            l.get::<T>(base.as_bytes()).ok()
        }
    }

    fn lib() -> Option<&'static Library> {
        static LIB: OnceLock<Option<Library>> = OnceLock::new();
        LIB.get_or_init(|| {
            for name in CANDIDATES {
                // SAFETY: see the libde265 `lib()` SAFETY comment.
                if let Ok(l) = unsafe { Library::new(name) } {
                    return Some(l);
                }
            }
            None
        })
        .as_ref()
    }

    /// True iff a libx265 shared library was successfully loaded AND
    /// the per-version `_NNN` symbols we need are present. Returns
    /// false if e.g. only an old libx265 (no `_216` suffix) ships.
    pub fn available() -> bool {
        let Some(l) = lib() else {
            return false;
        };
        type ParamAllocFn = unsafe extern "C" fn() -> *mut c_void;
        type EncoderEncodeFn = unsafe extern "C" fn(
            *mut c_void,
            *mut *mut c_void,
            *mut u32,
            *mut c_void,
            *mut c_void,
        ) -> i32;
        let pa: Option<Symbol<ParamAllocFn>> = lookup(l, "x265_param_alloc");
        let ee: Option<Symbol<EncoderEncodeFn>> = lookup(l, "x265_encoder_encode");
        pa.is_some() && ee.is_some()
    }

    // x265_picture prefix layout — stable since x265 1.x. See <x265.h>:
    //   off  0  i64  pts
    //   off  8  i64  dts
    //   off 16  i32  vbvEndFlag
    //   off 20      <4-byte padding for 8-byte alignment of next void*>
    //   off 24  *   userData
    //   off 32  * x4 planes[4]
    //   off 64  i32 x4 stride[4]
    //   off 80  i32 bitDepth
    //   off 84  i32 sliceType
    //   off 88  i32 poc
    //   off 92  i32 colorSpace
    // We never write past colorSpace; the rest of the struct stays
    // exactly as `x265_picture_init` left it.
    const OFF_PLANES: usize = 32;
    const OFF_STRIDE: usize = 64;
    const OFF_BIT_DEPTH: usize = 80;
    const OFF_COLOR_SPACE: usize = 92;
    // X265_CSP_I420 == 1 per the x265.h `X265_CSP_I420` enum entry
    // (chroma format ids match the libde265 ones).
    const X265_CSP_I420: i32 = 1;

    /// Encode an 8-bit 4:2:0 single-frame picture losslessly, returning
    /// the concatenated Annex-B NAL bytes (VPS+SPS+PPS+IDR slice plus
    /// any flush trailers x265 emits). Returns `None` if libx265 is
    /// unavailable or the encoder errored at any step.
    pub fn encode_lossless_yuv420(
        width: u32,
        height: u32,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
    ) -> Option<Vec<u8>> {
        type ParamAllocFn = unsafe extern "C" fn() -> *mut c_void;
        type ParamFreeFn = unsafe extern "C" fn(*mut c_void);
        type ParamDefaultPresetFn = unsafe extern "C" fn(*mut c_void, *const u8, *const u8) -> i32;
        type ParamParseFn = unsafe extern "C" fn(*mut c_void, *const u8, *const u8) -> i32;
        type EncoderOpenFn = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
        type EncoderEncodeFn = unsafe extern "C" fn(
            *mut c_void,
            *mut *mut u8,
            *mut u32,
            *mut c_void,
            *mut c_void,
        ) -> i32;
        type EncoderCloseFn = unsafe extern "C" fn(*mut c_void);
        type PictureAllocFn = unsafe extern "C" fn() -> *mut c_void;
        type PictureFreeFn = unsafe extern "C" fn(*mut c_void);
        type PictureInitFn = unsafe extern "C" fn(*mut c_void, *mut c_void);

        // Sanity: 4:2:0 needs even dims and matching chroma plane sizes.
        if width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 {
            return None;
        }
        let w = width as usize;
        let h = height as usize;
        if y.len() < w * h || cb.len() < (w / 2) * (h / 2) || cr.len() < (w / 2) * (h / 2) {
            return None;
        }

        let l = lib()?;
        unsafe {
            let param_alloc: Symbol<ParamAllocFn> = lookup(l, "x265_param_alloc")?;
            let param_free: Symbol<ParamFreeFn> = lookup(l, "x265_param_free")?;
            let param_default_preset: Symbol<ParamDefaultPresetFn> =
                lookup(l, "x265_param_default_preset")?;
            let param_parse: Symbol<ParamParseFn> = lookup(l, "x265_param_parse")?;
            let encoder_open: Symbol<EncoderOpenFn> = lookup(l, "x265_encoder_open")?;
            let encoder_encode: Symbol<EncoderEncodeFn> = lookup(l, "x265_encoder_encode")?;
            let encoder_close: Symbol<EncoderCloseFn> = lookup(l, "x265_encoder_close")?;
            let picture_alloc: Symbol<PictureAllocFn> = lookup(l, "x265_picture_alloc")?;
            let picture_free: Symbol<PictureFreeFn> = lookup(l, "x265_picture_free")?;
            let picture_init: Symbol<PictureInitFn> = lookup(l, "x265_picture_init")?;

            let param = param_alloc();
            if param.is_null() {
                return None;
            }
            // Wrap everything below in a closure so we can unwind to
            // the cleanup epilogue on early-out.
            let result = (|| -> Option<Vec<u8>> {
                let preset = CString::new("ultrafast").unwrap();
                let tune = CString::new("psnr").unwrap();
                if param_default_preset(
                    param,
                    preset.as_ptr() as *const u8,
                    tune.as_ptr() as *const u8,
                ) != 0
                {
                    return None;
                }
                // Set width/height/fps/csp/lossless via the string
                // interface so we don't have to mirror the giant
                // x265_param struct. `x265_param_parse` returns 0 on
                // success.
                let pairs: &[(&[u8], String)] = &[
                    (b"input-csp\0", "i420".to_string()),
                    (b"input-depth\0", "8".to_string()),
                    (b"output-depth\0", "8".to_string()),
                    (b"input-res\0", format!("{width}x{height}")),
                    (b"fps\0", "30".to_string()),
                    (b"lossless\0", "1".to_string()),
                    // Single-frame stream — no lookahead, no B-frames.
                    (b"keyint\0", "1".to_string()),
                    (b"min-keyint\0", "1".to_string()),
                    (b"bframes\0", "0".to_string()),
                    (b"frame-threads\0", "1".to_string()),
                    (b"pools\0", "none".to_string()),
                    (b"log-level\0", "none".to_string()),
                    (b"repeat-headers\0", "1".to_string()),
                    (b"annexb\0", "1".to_string()),
                ];
                for (name, value) in pairs {
                    let c_value = CString::new(value.as_str()).ok()?;
                    let rc = param_parse(param, name.as_ptr(), c_value.as_ptr() as *const u8);
                    // x265_param_parse returns negative on hard error,
                    // 0 on success, positive for "ignored / clamped".
                    if rc < 0 {
                        return None;
                    }
                }

                let encoder = encoder_open(param);
                if encoder.is_null() {
                    return None;
                }

                let pic = picture_alloc();
                if pic.is_null() {
                    encoder_close(encoder);
                    return None;
                }
                picture_init(param, pic);

                // Write the prefix fields we control (planes, stride,
                // bitDepth, colorSpace) by byte offset — see the
                // OFF_* constants above.
                let pic_bytes = pic as *mut u8;
                let planes_ptr = pic_bytes.add(OFF_PLANES) as *mut *const u8;
                planes_ptr.write_unaligned(y.as_ptr());
                planes_ptr.add(1).write_unaligned(cb.as_ptr());
                planes_ptr.add(2).write_unaligned(cr.as_ptr());
                planes_ptr.add(3).write_unaligned(std::ptr::null());
                let stride_ptr = pic_bytes.add(OFF_STRIDE) as *mut i32;
                stride_ptr.write_unaligned(width as i32);
                stride_ptr.add(1).write_unaligned((width / 2) as i32);
                stride_ptr.add(2).write_unaligned((width / 2) as i32);
                stride_ptr.add(3).write_unaligned(0);
                let bit_depth_ptr = pic_bytes.add(OFF_BIT_DEPTH) as *mut i32;
                bit_depth_ptr.write_unaligned(8);
                let color_space_ptr = pic_bytes.add(OFF_COLOR_SPACE) as *mut i32;
                color_space_ptr.write_unaligned(X265_CSP_I420);

                let mut out = Vec::new();
                let mut nal_ptr: *mut u8 = std::ptr::null_mut();
                let mut nal_count: u32 = 0;
                // Send the one frame.
                let rc = encoder_encode(
                    encoder,
                    &mut nal_ptr as *mut *mut u8,
                    &mut nal_count,
                    pic,
                    std::ptr::null_mut(),
                );
                if rc < 0 {
                    picture_free(pic);
                    encoder_close(encoder);
                    return None;
                }
                append_nals(&mut out, nal_ptr, nal_count);

                // Drain the encoder until it returns 0 NALs (no more
                // pending output). Cap defensively.
                for _ in 0..16 {
                    nal_ptr = std::ptr::null_mut();
                    nal_count = 0;
                    let rc = encoder_encode(
                        encoder,
                        &mut nal_ptr as *mut *mut u8,
                        &mut nal_count,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    );
                    if rc < 0 {
                        break;
                    }
                    if nal_count == 0 {
                        break;
                    }
                    append_nals(&mut out, nal_ptr, nal_count);
                }

                picture_free(pic);
                encoder_close(encoder);

                if out.is_empty() {
                    None
                } else {
                    Some(out)
                }
            })();
            param_free(param);
            result
        }
    }

    /// Append the `nal_count` x265_nal payloads at `nal_ptr` to `out`.
    /// `x265_nal` layout (stable since 1.x): `{ u32 type; u32 sizeBytes;
    /// u8* payload; }` — total size 16 bytes (4 + 4 + 8) on 64-bit.
    /// We read each by offset to avoid mirroring the typedef.
    unsafe fn append_nals(out: &mut Vec<u8>, nal_ptr: *mut u8, nal_count: u32) {
        if nal_ptr.is_null() || nal_count == 0 {
            return;
        }
        const NAL_STRIDE: usize = 16;
        for i in 0..nal_count as usize {
            let entry = nal_ptr.add(i * NAL_STRIDE);
            let size_bytes = (entry.add(4) as *const u32).read_unaligned() as usize;
            let payload = (entry.add(8) as *const *const u8).read_unaligned();
            if !payload.is_null() && size_bytes > 0 {
                let slice = std::slice::from_raw_parts(payload, size_bytes);
                out.extend_from_slice(slice);
            }
        }
    }
}
