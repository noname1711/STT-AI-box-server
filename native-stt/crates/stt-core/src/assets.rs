use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};

#[derive(Debug, Clone)]
pub struct OfflineModelAssets {
    pub model_dir: PathBuf,
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub joiner: PathBuf,
    pub tokens: PathBuf,
    pub bpe_vocab: Option<PathBuf>,
    pub model_type: String,
}

fn resolve_model_dir(model_dir: &Path) -> Result<PathBuf> {
    let resolved = model_dir.canonicalize().with_context(|| {
        format!(
            "STT model directory does not exist: {}",
            model_dir.display()
        )
    })?;
    if !resolved.is_dir() {
        return Err(anyhow!(
            "STT model directory is not a directory: {}",
            resolved.display()
        ));
    }
    Ok(resolved)
}

fn path_in_model_dir(model_dir: &Path, path: &Path) -> Result<PathBuf> {
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        model_dir.join(path)
    };
    candidate
        .canonicalize()
        .with_context(|| format!("Missing asset: {}", candidate.display()))
}

fn find_with_patterns(
    model_dir: &Path,
    exact_names: &[&str],
    prefixes_and_suffixes: &[(&str, &str)],
) -> Result<PathBuf> {
    for name in exact_names {
        let candidate = model_dir.join(name);
        if candidate.is_file() {
            return candidate.canonicalize().map_err(Into::into);
        }
    }

    let mut matches = fs::read_dir(model_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| {
            path.file_name()
                .and_then(|file| file.to_str())
                .map(|file| {
                    prefixes_and_suffixes
                        .iter()
                        .any(|(prefix, suffix)| file.starts_with(prefix) && file.ends_with(suffix))
                })
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    matches.sort();

    matches
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("Missing required asset in {}", model_dir.display()))
}

fn find_optional_with_patterns(model_dir: &Path, exact_names: &[&str]) -> Result<Option<PathBuf>> {
    for name in exact_names {
        let candidate = model_dir.join(name);
        if candidate.is_file() {
            return candidate.canonicalize().map(Some).map_err(Into::into);
        }
    }
    Ok(None)
}

pub fn resolve_offline_model_assets(
    model_dir: &Path,
    model_type: &str,
    encoder: Option<&Path>,
    decoder: Option<&Path>,
    joiner: Option<&Path>,
    tokens: Option<&Path>,
    bpe_vocab: Option<&Path>,
) -> Result<OfflineModelAssets> {
    let model_dir = resolve_model_dir(model_dir)?;

    let encoder = match encoder {
        Some(path) => path_in_model_dir(&model_dir, path)?,
        None => find_with_patterns(
            &model_dir,
            &["encoder.int8.onnx", "encoder.onnx"],
            &[("encoder-", ".int8.onnx"), ("encoder-", ".onnx")],
        )?,
    };
    let decoder = match decoder {
        Some(path) => path_in_model_dir(&model_dir, path)?,
        None => find_with_patterns(
            &model_dir,
            &["decoder.onnx", "decoder.int8.onnx"],
            &[("decoder-", ".onnx"), ("decoder-", ".int8.onnx")],
        )?,
    };
    let joiner = match joiner {
        Some(path) => path_in_model_dir(&model_dir, path)?,
        None => find_with_patterns(
            &model_dir,
            &["joiner.int8.onnx", "joiner.onnx"],
            &[("joiner-", ".int8.onnx"), ("joiner-", ".onnx")],
        )?,
    };
    let tokens = match tokens {
        Some(path) => path_in_model_dir(&model_dir, path)?,
        None => find_with_patterns(&model_dir, &["tokens.txt", "config.json"], &[])?,
    };
    let bpe_vocab = match bpe_vocab {
        Some(path) => Some(path_in_model_dir(&model_dir, path)?),
        // bpe_vocab is a sherpa contextual-bias vocabulary text file.
        // Never auto-discover SentencePiece's binary bpe.model here.
        None => None,
    };

    Ok(OfflineModelAssets {
        model_dir,
        encoder,
        decoder,
        joiner,
        tokens,
        bpe_vocab,
        model_type: model_type.to_string(),
    })
}
