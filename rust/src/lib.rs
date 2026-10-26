//! Pocket TTS as a library, beside the binary.
//!
//! The crate shipped only a `[[bin]]`. Everything the CLI does — reading a
//! language config, resolving weights from the Hub, loading the model,
//! generating audio frame by frame — was already written here and already
//! private. This file does nothing but make it callable: the modules below are
//! the same ones `main.rs` declared, with `pub` in front.
//!
//! # Why that is worth a change rather than a dependency on `ptts`
//!
//! `ptts` is the model; this crate is what makes **pocket-tts checkpoints**
//! work on it. The difference is not decoration. Measured against stock
//! `ptts` 0.5.0: a pocket-tts French checkpoint fails to load over
//! `flow_lm.bos_before_voice`, and with that tensor removed it produces noise
//! of a fixed length — the same for every sentence. The pinned `ptts` fork this
//! crate depends on is what closes that gap, together with the config
//! translation in [`config`] and the weight resolution in [`weights`].
//!
//! So a caller that wants Kyutai's voices in its own process wants *this*
//! crate, not the one underneath it.
//!
//! # What it costs to link, and what was done about it
//!
//! Two things stood in the way, and both are features now rather than
//! obligations:
//!
//! - **SentencePiece dragged CMake in.** It is needed by the tokenizers of most
//!   community models and by none of the released ones, which ship a Hugging
//!   Face `tokenizer.json`. It now sits behind the `sentencepiece` feature, off
//!   by default, so the library builds on a machine with no C++ toolchain.
//! - **The server and the CLI dragged in axum, clap, tokio, tower-http and
//!   tempfile.** None of the six is used by anything below the `cli` feature,
//!   which the binary requires and a library consumer turns off.
//!
//! ```toml
//! pocket-tts = { git = "…", default-features = false }
//! ```

pub mod audio;
pub mod config;
pub mod defaults;
pub mod download;
pub mod model;
pub mod rng;
pub mod text;
pub mod tokenizer;
pub mod weights;

#[cfg(feature = "cli")]
pub mod logging;
#[cfg(feature = "cli")]
pub mod server;

/// The f32 CPU backend, which is what `generate` and `serve` use without
/// `--quantize`.
///
/// Re-exported as an alias so a caller can name [`model::TtsModel`]'s type
/// parameter without depending on `xn` itself — the crate whose version has to
/// match the one `ptts` uses, which is exactly the kind of coupling a library
/// should absorb rather than pass on.
pub type F32 = xn::Unquantized<f32, xn::CpuDevice>;

/// The int8 backend, which `--quantize` selects: less memory, a slightly
/// different sound.
pub type Int8 = xn::quantized::Q80F32;
