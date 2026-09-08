#![cfg(feature = "onnx-spike")]

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use ort::session::builder::GraphOptimizationLevel;
use ort::{inputs, session::Session, value::Tensor};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use stt_capu::build_capu_postprocessor;
use stt_core::config::WorkspacePaths;
use stt_core::postprocess::{
    BuiltinPostprocessor, PostprocessMode, Postprocessor, clean_text, finalize_text,
};
use stt_core::runtime::SttRuntime;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(version, about = "CAPU ONNX + pure Rust spike")]
struct Cli {
    #[arg(long, default_value = ".")]
    workspace_root: PathBuf,
    #[arg(long, default_value = "target/capu-generated/vibert-capu-onnx")]
    export_dir: PathBuf,
    #[arg(long, default_value_t = 1)]
    ort_intra_threads: usize,
    #[arg(long, default_value_t = false)]
    ort_memory_pattern: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Process a single text through Rust ONNX CAPU pipeline.
    Run { text: String },
    /// Benchmark Rust ONNX vs Python worker on known-reference snippets.
    Benchmark {
        #[arg(long, default_value = "baselines/phase0/capu_snippets.jsonl")]
        snippets: PathBuf,
        #[arg(long, default_value = "model_card_example")]
        benchmark_id: String,
        #[arg(long, default_value_t = 100)]
        iterations: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Benchmark across audio-derived transcripts of varying lengths.
    BenchmarkAssets {
        #[arg(long, default_value = "vit_stt_vi_v2")]
        model_id: String,
        #[arg(
            long,
            default_value = "/Users/leakless/code/sherpa-onnx-vit/tests/assets"
        )]
        assets_dir: PathBuf,
        #[arg(long, default_value_t = 10)]
        iterations: usize,
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

// ---------------------------------------------------------------------------
// Export metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct ExportMetadata {
    #[allow(dead_code)]
    start_token: String,
    #[allow(dead_code)]
    start_token_id: i64,
    max_len: usize,
    min_len: usize,
    iterations: usize,
    min_error_probability: f32,
    split_chunk: bool,
    chunk_size: usize,
    overlap_size: usize,
    min_words_cut: usize,
    punctuation: Vec<String>,
    static_token_len: Option<usize>,
    static_word_len: Option<usize>,
    model_file: Option<String>,
}

#[derive(Debug, Clone)]
struct ChunkConfig {
    split_chunk: bool,
    chunk_size: usize,
    overlap_size: usize,
    min_words_cut: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct OrtTuning {
    intra_threads: usize,
    memory_pattern: bool,
}

type ChunkIndices = Vec<(usize, usize)>;
type PreparedBatch = (Vec<Vec<String>>, Option<ChunkIndices>);
type PredictOutput = (Vec<Vec<f32>>, Vec<Vec<usize>>, Vec<f32>);

// ---------------------------------------------------------------------------
// Vocabulary helpers (unchanged from original Python port)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CapuVocab {
    labels: Vec<String>,
    noop_index: usize,
    incorr_index: usize,
    verb_decode: HashMap<String, String>,
}

impl CapuVocab {
    fn load(export_dir: &Path) -> Result<Self> {
        let labels = read_lines(&export_dir.join("labels.txt"))?;
        let d_tags = read_lines(&export_dir.join("d_tags.txt"))?;
        let noop_index = labels
            .iter()
            .position(|label| label == "$KEEP")
            .context("$KEEP missing from labels.txt")?;
        let incorr_index = d_tags
            .iter()
            .position(|tag| tag == "INCORRECT")
            .context("INCORRECT missing from d_tags.txt")?;
        Ok(Self {
            labels,
            noop_index,
            incorr_index,
            verb_decode: read_verb_decode_map(&export_dir.join("verb-form-vocab.txt"))?,
        })
    }

    fn label(&self, index: usize) -> Result<&str> {
        self.labels
            .get(index)
            .map(String::as_str)
            .with_context(|| format!("label index out of range: {index}"))
    }
}

#[derive(Debug, Clone)]
struct SequenceInputs {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    input_offsets: Vec<i64>,
}

// ---------------------------------------------------------------------------
// Stage-level timing accumulator
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize)]
struct StageTimings {
    /// Time spent cleaning input text (clean_text).
    clean_ms: f64,
    /// Time spent splitting text into chunks (split_chunks).
    split_chunk_ms: f64,
    /// Time spent tokenizing (HF tokenizer encode per iteration).
    tokenize_ms: f64,
    /// Time spent in ORT session.run.
    ort_ms: f64,
    /// Time spent in softmax + argmax + convert_outputs.
    softmax_ms: f64,
    /// Time spent applying edits (postprocess_batch + update_final_batch).
    edit_apply_ms: f64,
    /// Time spent merging overlapping chunks.
    merge_ms: f64,
    /// Time spent in finalize_text.
    finalize_ms: f64,
    /// Wall-clock total (sum of all stages, should match wall clock).
    total_ms: f64,
}

// ---------------------------------------------------------------------------
// ONNX CAPU runner
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct CapuOnnxRunner {
    session: Session,
    tokenizer: Tokenizer,
    vocab: CapuVocab,
    max_len: usize,
    min_len: usize,
    iterations: usize,
    min_error_probability: f32,
    chunk: ChunkConfig,
    punctuation: Vec<String>,
    static_token_len: Option<usize>,
    static_word_len: Option<usize>,
}

impl CapuOnnxRunner {
    fn load(export_dir: &Path, tuning: OrtTuning) -> Result<Self> {
        let metadata: ExportMetadata = serde_json::from_str(
            &fs::read_to_string(export_dir.join("export-metadata.json")).with_context(|| {
                format!("failed reading export metadata in {}", export_dir.display())
            })?,
        )?;

        let mut session_builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("ORT session builder init failed: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("ORT optimization level failed: {e}"))?;
        session_builder = session_builder
            .with_intra_threads(tuning.intra_threads)
            .map_err(|e| anyhow::anyhow!("ORT intra threads failed: {e}"))?;
        session_builder = session_builder
            .with_memory_pattern(tuning.memory_pattern)
            .map_err(|e| anyhow::anyhow!("ORT memory pattern failed: {e}"))?;

        let model_file = metadata.model_file.as_deref().unwrap_or("seq2labels.onnx");
        let session = session_builder
            .commit_from_file(export_dir.join(model_file))
            .map_err(|e| anyhow::anyhow!("ORT commit model failed: {e}"))?;

        // Load HuggingFace fast tokenizer from the exported ONNX bundle (req #1)
        let tokenizer = Tokenizer::from_file(export_dir.join("tokenizer.json")).map_err(|e| {
            anyhow::anyhow!(
                "failed loading tokenizer.json from {}: {e}",
                export_dir.display()
            )
        })?;

        let vocab = CapuVocab::load(export_dir)?;

        Ok(Self {
            session,
            tokenizer,
            vocab,
            max_len: metadata.max_len,
            min_len: metadata.min_len,
            iterations: metadata.iterations,
            min_error_probability: metadata.min_error_probability,
            chunk: ChunkConfig {
                split_chunk: metadata.split_chunk,
                chunk_size: metadata.chunk_size,
                overlap_size: metadata.overlap_size,
                min_words_cut: metadata.min_words_cut,
            },
            punctuation: metadata.punctuation,
            static_token_len: metadata.static_token_len,
            static_word_len: metadata.static_word_len,
        })
    }

    /// Process text without detailed stage timing (keeps existing API working).
    fn process_text(&mut self, text: &str) -> Result<String> {
        let cleaned = clean_text(text);
        if cleaned.is_empty() {
            return Ok(cleaned);
        }
        let output = self.forward_single(&cleaned)?;
        Ok(finalize_text(&output))
    }

    /// Process text with per-stage timing collection.
    fn process_text_with_timings(&mut self, text: &str) -> Result<(String, StageTimings)> {
        let total = Instant::now();
        let mut t = StageTimings::default();

        let t_clean = Instant::now();
        let cleaned = clean_text(text);
        t.clean_ms = ms(t_clean);
        if cleaned.is_empty() {
            t.total_ms = ms(total);
            return Ok((cleaned, t));
        }

        let (output, inner_t) = self.forward_single_timed(&cleaned)?;

        let t_final = Instant::now();
        let result = finalize_text(&output);
        t.finalize_ms = ms(t_final);

        t.split_chunk_ms = inner_t.split_chunk_ms;
        t.tokenize_ms = inner_t.tokenize_ms;
        t.ort_ms = inner_t.ort_ms;
        t.softmax_ms = inner_t.softmax_ms;
        t.edit_apply_ms = inner_t.edit_apply_ms;
        t.merge_ms = inner_t.merge_ms;
        t.total_ms = ms(total);

        Ok((result, t))
    }

    // -----------------------------------------------------------------------
    // forward_single (without timing — keeps existing callers unchanged)
    // -----------------------------------------------------------------------

    fn forward_single(&mut self, text: &str) -> Result<String> {
        let tokens = text
            .split_whitespace()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();

        let (final_batch, indices) = self.prepare_batch(tokens);
        self.run_iterations(final_batch, &indices)
    }

    // -----------------------------------------------------------------------
    // forward_single with stage timings
    // -----------------------------------------------------------------------

    fn forward_single_timed(&mut self, text: &str) -> Result<(String, StageTimings)> {
        let mut t = StageTimings::default();

        let t_split = Instant::now();
        let tokens = text
            .split_whitespace()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        let (final_batch, indices) = self.prepare_batch(tokens);
        t.split_chunk_ms = ms(t_split);

        let (merged, inner_t) = self.run_iterations_timed(final_batch, &indices)?;
        t.tokenize_ms = inner_t.tokenize_ms;
        t.ort_ms = inner_t.ort_ms;
        t.softmax_ms = inner_t.softmax_ms;
        t.edit_apply_ms = inner_t.edit_apply_ms;
        t.merge_ms = inner_t.merge_ms;

        Ok((merged, t))
    }

    /// Split the word list into chunks if needed, returning (batch, indices).
    fn prepare_batch(&self, tokens: Vec<String>) -> PreparedBatch {
        let mut batch = vec![tokens];
        let indices = if self.chunk.split_chunk {
            let (split, indices) = self.split_chunks(&batch);
            batch = split;
            Some(indices)
        } else {
            None
        };
        (batch, indices)
    }

    /// Run iterative CAPU refinement without timing.
    fn run_iterations(
        &mut self,
        final_batch: Vec<Vec<String>>,
        indices: &Option<Vec<(usize, usize)>>,
    ) -> Result<String> {
        let (merged, _) = self.run_loop(final_batch, indices, None)?;
        Ok(merged)
    }

    /// Run iterative CAPU refinement with timing.
    fn run_iterations_timed(
        &mut self,
        final_batch: Vec<Vec<String>>,
        indices: &Option<Vec<(usize, usize)>>,
    ) -> Result<(String, StageTimings)> {
        let mut t = StageTimings::default();
        let (merged, _) = self.run_loop(final_batch, indices, Some(&mut t))?;
        Ok((merged, t))
    }

    /// Shared iteration loop; optional `timings` accumulates per-iteration sub-stages.
    fn run_loop(
        &mut self,
        mut final_batch: Vec<Vec<String>>,
        indices: &Option<Vec<(usize, usize)>>,
        timings: Option<&mut StageTimings>,
    ) -> Result<(String, StageTimings)> {
        let mut local_t = StageTimings::default();
        let t = timings.unwrap_or(&mut local_t);

        let short_ids: Vec<usize> = (0..final_batch.len())
            .filter(|&i| final_batch[i].len() < self.min_len)
            .collect();
        let mut pred_ids: Vec<usize> = (0..final_batch.len())
            .filter(|i| !short_ids.contains(i))
            .collect();
        let mut prev_preds: HashMap<usize, Vec<Vec<String>>> = final_batch
            .iter()
            .enumerate()
            .map(|(idx, item)| (idx, vec![item.clone()]))
            .collect();

        for _ in 0..self.iterations {
            let orig_batch: Vec<Vec<String>> = pred_ids
                .iter()
                .map(|&idx| final_batch[idx].clone())
                .collect();
            if orig_batch.is_empty() {
                break;
            }

            // tokenize_ms
            let t_tok = Instant::now();
            let sequences = self.preprocess(&orig_batch);
            t.tokenize_ms += ms(t_tok);
            if sequences.is_empty() {
                break;
            }

            // ORT run + softmax combined
            let t_ort = Instant::now();
            let (probabilities, idxs, error_probs) = self.predict(&sequences)?;
            t.ort_ms += ms(t_ort);

            // edit apply
            let t_edit = Instant::now();
            let pred_batch =
                self.postprocess_batch(&orig_batch, &probabilities, &idxs, &error_probs)?;
            let (new_final, new_pred_ids, _) =
                self.update_final_batch(final_batch, &pred_ids, pred_batch, &mut prev_preds);
            final_batch = new_final;
            pred_ids = new_pred_ids;
            t.edit_apply_ms += ms(t_edit);

            if pred_ids.is_empty() {
                break;
            }
        }

        let t_merge = Instant::now();
        let merged = if let Some(indices) = indices {
            indices
                .iter()
                .map(|&(start, end)| self.merge_chunks(&final_batch[start..end]))
                .collect::<Vec<_>>()
        } else {
            final_batch
                .into_iter()
                .map(|tokens| tokens.join(" "))
                .collect::<Vec<_>>()
        };
        let mut merged_result = merged.join(" ");
        merged_result = strip_spaces_before_punctuation(&merged_result, &self.punctuation);
        t.merge_ms += ms(t_merge);

        Ok((merged_result, std::mem::take(&mut local_t)))
    }

    // -----------------------------------------------------------------------
    // Tokenization (replaces hand-written WordPieceTokenizer — req #1)
    // -----------------------------------------------------------------------

    fn preprocess(&self, batch: &[Vec<String>]) -> Vec<SequenceInputs> {
        batch
            .iter()
            .filter(|sequence| !sequence.is_empty())
            .map(|sequence| {
                // Build pretokenized word list: START + content (truncated to max_len)
                let cap = sequence.len().min(self.max_len) + 1; // +1 for $START
                let mut words: Vec<&str> = Vec::with_capacity(cap);
                words.push("$START");
                for w in sequence.iter().take(self.max_len) {
                    words.push(w.as_str());
                }

                // Encode with HuggingFace fast tokenizer.
                // add_special_tokens=false because we manually prepend $START.
                // The &[&str] input triggers InputSequence::PreTokenized, matching
                // Python's is_split_into_words=True behavior.
                let encoding = self
                    .tokenizer
                    .encode(&words[..], false)
                    .expect("HF tokenizer encode should not fail");

                let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
                let attention_mask: Vec<i64> = encoding
                    .get_attention_mask()
                    .iter()
                    .map(|&m| m as i64)
                    .collect();

                // Compute input_offsets from word_ids transitions — exactly like
                // Python's GecBERTModel.preprocess.
                let word_ids = encoding.get_word_ids();
                let mut input_offsets = vec![0i64];
                for i in 1..word_ids.len() {
                    if word_ids[i] != word_ids[i - 1] {
                        input_offsets.push(i as i64);
                    }
                }

                SequenceInputs {
                    input_ids,
                    attention_mask,
                    input_offsets,
                }
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // ONNX predict (timing already instrumented at call site)
    // -----------------------------------------------------------------------

    fn predict(&mut self, sequences: &[SequenceInputs]) -> Result<PredictOutput> {
        let batch_size = sequences.len();
        let observed_token_len = sequences
            .iter()
            .map(|item| item.input_ids.len())
            .max()
            .unwrap_or(0);
        let observed_word_len = sequences
            .iter()
            .map(|item| item.input_offsets.len())
            .max()
            .unwrap_or(0);
        let token_len = self.static_token_len.unwrap_or(observed_token_len);
        let word_len = self.static_word_len.unwrap_or(observed_word_len);
        if observed_token_len > token_len {
            bail!(
                "token sequence length {observed_token_len} exceeds static ONNX token length {token_len}"
            );
        }
        if observed_word_len > word_len {
            bail!(
                "word sequence length {observed_word_len} exceeds static ONNX word length {word_len}"
            );
        }

        let mut input_ids = vec![0_i64; batch_size * token_len];
        let mut attention_mask = vec![0_i64; batch_size * token_len];
        let mut input_offsets = vec![0_i64; batch_size * word_len];

        for (batch_index, sequence) in sequences.iter().enumerate() {
            for (index, value) in sequence.input_ids.iter().enumerate() {
                input_ids[batch_index * token_len + index] = *value;
            }
            for (index, value) in sequence.attention_mask.iter().enumerate() {
                attention_mask[batch_index * token_len + index] = *value;
            }
            for (index, value) in sequence.input_offsets.iter().enumerate() {
                input_offsets[batch_index * word_len + index] = *value;
            }
        }

        let (logits_shape, logits, detect_logits_shape, detect_logits) = {
            let outputs = self.session.run(inputs![
                "input_ids" => Tensor::<i64>::from_array((vec![batch_size, token_len], input_ids))?,
                "attention_mask" => Tensor::<i64>::from_array((vec![batch_size, token_len], attention_mask))?,
                "input_offsets" => Tensor::<i64>::from_array((vec![batch_size, word_len], input_offsets))?,
            ])?;

            let (logits_shape, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
            let (detect_logits_shape, detect_logits) =
                outputs["detect_logits"].try_extract_tensor::<f32>()?;
            (
                logits_shape.iter().copied().collect::<Vec<_>>(),
                logits.to_vec(),
                detect_logits_shape.iter().copied().collect::<Vec<_>>(),
                detect_logits.to_vec(),
            )
        };

        self.convert_outputs(
            &logits_shape,
            &logits,
            &detect_logits_shape,
            &detect_logits,
            sequences,
        )
    }

    fn convert_outputs(
        &self,
        logits_shape: &[i64],
        logits: &[f32],
        detect_logits_shape: &[i64],
        detect_logits: &[f32],
        sequences: &[SequenceInputs],
    ) -> Result<PredictOutput> {
        if logits_shape.len() != 3 || detect_logits_shape.len() != 3 {
            bail!("expected rank-3 logits and detect_logits outputs");
        }
        let logits_seq_len = logits_shape[1] as usize;
        let logits_classes = logits_shape[2] as usize;
        let detect_seq_len = detect_logits_shape[1] as usize;
        let detect_classes = detect_logits_shape[2] as usize;

        let mut probabilities = Vec::with_capacity(sequences.len());
        let mut indices = Vec::with_capacity(sequences.len());
        let mut error_probs = Vec::with_capacity(sequences.len());

        for (batch_index, sequence) in sequences.iter().enumerate() {
            let row_len = sequence.input_offsets.len();
            let mut row_probs = Vec::with_capacity(row_len);
            let mut row_indices = Vec::with_capacity(row_len);
            let mut max_error_prob = 0.0_f32;

            for position in 0..row_len {
                let label_offset = (batch_index * logits_seq_len + position) * logits_classes;
                let label_probs = softmax(&logits[label_offset..label_offset + logits_classes]);
                let (label_index, label_prob) = argmax(&label_probs);
                row_probs.push(label_prob);
                row_indices.push(label_index);

                let detect_offset = (batch_index * detect_seq_len + position) * detect_classes;
                let detect_probs =
                    softmax(&detect_logits[detect_offset..detect_offset + detect_classes]);
                let incorr_prob = *detect_probs
                    .get(self.vocab.incorr_index)
                    .context("detect INCORRECT index out of range")?;
                if incorr_prob > max_error_prob {
                    max_error_prob = incorr_prob;
                }
            }

            probabilities.push(row_probs);
            indices.push(row_indices);
            error_probs.push(max_error_prob);
        }

        Ok((probabilities, indices, error_probs))
    }

    // -----------------------------------------------------------------------
    // Post-processing (unchanged from original)
    // -----------------------------------------------------------------------

    fn postprocess_batch(
        &self,
        batch: &[Vec<String>],
        probabilities: &[Vec<f32>],
        indices: &[Vec<usize>],
        error_probs: &[f32],
    ) -> Result<Vec<Vec<String>>> {
        let mut results = Vec::with_capacity(batch.len());
        for ((tokens, probs), (idxs, error_prob)) in batch
            .iter()
            .zip(probabilities)
            .zip(indices.iter().zip(error_probs))
        {
            let length = tokens.len().min(self.max_len);
            if idxs
                .iter()
                .take(length + 1)
                .all(|value| *value == self.vocab.noop_index)
                || *error_prob < self.min_error_probability
            {
                results.push(tokens.clone());
                continue;
            }

            let mut edits = Vec::new();
            for i in 0..=length {
                if i >= idxs.len() || i >= probs.len() {
                    break;
                }
                if idxs[i] == self.vocab.noop_index {
                    continue;
                }
                let token = if i == 0 {
                    "$START"
                } else {
                    tokens[i - 1].as_str()
                };
                let suggestion = self.vocab.label(idxs[i])?;
                if let Some(action) = get_token_action(token, i, probs[i], suggestion) {
                    edits.push(action);
                }
            }
            results.push(apply_edits(tokens, &edits, &self.vocab.verb_decode));
        }
        Ok(results)
    }

    fn update_final_batch(
        &self,
        mut final_batch: Vec<Vec<String>>,
        pred_ids: &[usize],
        pred_batch: Vec<Vec<String>>,
        prev_preds: &mut HashMap<usize, Vec<Vec<String>>>,
    ) -> (Vec<Vec<String>>, Vec<usize>, usize) {
        let mut new_pred_ids = Vec::new();
        let mut total_updated = 0;

        for (offset, &orig_id) in pred_ids.iter().enumerate() {
            let orig = final_batch[orig_id].clone();
            let pred = pred_batch[offset].clone();
            let history = prev_preds.entry(orig_id).or_default();
            if orig != pred && !history.contains(&pred) {
                final_batch[orig_id] = pred.clone();
                history.push(pred);
                new_pred_ids.push(orig_id);
                total_updated += 1;
            } else if orig != pred {
                final_batch[orig_id] = pred;
                total_updated += 1;
            }
        }

        (final_batch, new_pred_ids, total_updated)
    }

    fn split_chunks(&self, batch: &[Vec<String>]) -> (Vec<Vec<String>>, Vec<(usize, usize)>) {
        let mut result = Vec::new();
        let mut indices = Vec::with_capacity(batch.len());
        for tokens in batch {
            let start = result.len();
            let num_tokens = tokens.len();
            if num_tokens <= self.chunk.chunk_size {
                result.push(tokens.clone());
            } else if num_tokens < (self.chunk.chunk_size * 2 - self.chunk.overlap_size) {
                let split_idx = (num_tokens + self.chunk.overlap_size).div_ceil(2);
                result.push(tokens[..split_idx].to_vec());
                result.push(tokens[split_idx - self.chunk.overlap_size..].to_vec());
            } else {
                let stride = self.chunk.chunk_size - self.chunk.overlap_size;
                let mut i = 0;
                while i < num_tokens - self.chunk.overlap_size {
                    let end = (i + self.chunk.chunk_size).min(num_tokens);
                    result.push(tokens[i..end].to_vec());
                    i += stride;
                }
            }
            indices.push((start, result.len()));
        }
        (result, indices)
    }

    fn merge_chunks(&self, batch: &[Vec<String>]) -> String {
        if batch.len() <= 1 || self.chunk.overlap_size == 0 {
            return batch
                .iter()
                .flat_map(|chunk| chunk.iter().cloned())
                .collect::<Vec<_>>()
                .join(" ");
        }

        let mut result: Vec<String> = Vec::new();
        for sub_tokens in batch {
            result = self.apply_chunk_merging(&result, sub_tokens);
        }
        result.join(" ")
    }

    fn apply_chunk_merging(&self, tokens: &[String], next_tokens: &[String]) -> Vec<String> {
        if tokens.is_empty() {
            return next_tokens.to_vec();
        }

        let mut source_token_idx = Vec::new();
        let mut target_token_idx = Vec::new();
        let mut source_tokens = Vec::new();
        let mut target_tokens = Vec::new();
        let mut num_keep = self
            .chunk
            .overlap_size
            .saturating_sub(self.chunk.min_words_cut);

        let mut i = tokens.len() as isize;
        while source_token_idx.len() < self.chunk.overlap_size && i > 0 {
            i -= 1;
            let index = i as usize;
            if !self.is_punctuation(&tokens[index]) {
                source_token_idx.insert(0, index);
                source_tokens.insert(0, tokens[index].to_lowercase());
            }
        }

        let mut i = 0_usize;
        while target_token_idx.len() < self.chunk.overlap_size && i < next_tokens.len() {
            if !self.is_punctuation(&next_tokens[i]) {
                target_token_idx.push(i);
                target_tokens.push(next_tokens[i].to_lowercase());
            }
            i += 1;
        }

        let (mut tail_idx, mut head_idx) = (tokens.len(), 0_usize);
        for opcode in sequence_matcher(&source_tokens, &target_tokens) {
            match opcode.tag.as_str() {
                "equal" => {
                    if opcode.i1 >= num_keep {
                        tail_idx = source_token_idx[opcode.i1];
                        head_idx = target_token_idx[opcode.j1];
                        break;
                    }
                    if opcode.i2 > num_keep {
                        let source_keep_index = num_keep;
                        let target_keep_index = python_index(
                            target_token_idx.len(),
                            opcode.j2 as isize - opcode.i2 as isize + num_keep as isize,
                        )
                        .unwrap_or(0);
                        tail_idx = source_token_idx[source_keep_index];
                        head_idx = target_token_idx[target_keep_index];
                        break;
                    }
                }
                "delete" if opcode.i1 == 0 => {
                    num_keep += (opcode.i2 - opcode.i1) / 2;
                }
                _ => {}
            }
        }

        let mut merged = tokens[..tail_idx].to_vec();
        merged.extend_from_slice(&next_tokens[head_idx..]);
        merged
    }

    fn is_punctuation(&self, token: &str) -> bool {
        self.punctuation.iter().any(|punct| punct == token)
    }
}

// ---------------------------------------------------------------------------
// Edit actions (unchanged from original)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct EditAction {
    start: isize,
    end: isize,
    label: String,
}

fn get_token_action(token: &str, index: usize, prob: f32, suggestion: &str) -> Option<EditAction> {
    if prob < 0.0 || matches!(suggestion, "@@UNKNOWN@@" | "@@PADDING@@" | "$KEEP") {
        return None;
    }

    let (start_pos, end_pos) = if suggestion.starts_with("$REPLACE_")
        || suggestion.starts_with("$TRANSFORM_")
        || suggestion == "$DELETE"
    {
        (index as isize, index as isize + 1)
    } else if suggestion.starts_with("$APPEND_") || suggestion.starts_with("$MERGE_") {
        (index as isize + 1, index as isize + 1)
    } else {
        return None;
    };

    let cleaned = if suggestion == "$DELETE" {
        String::new()
    } else if suggestion.starts_with("$TRANSFORM_") || suggestion.starts_with("$MERGE_") {
        suggestion.to_string()
    } else {
        suggestion
            .split_once('_')
            .map(|(_, value)| value.to_string())
            .unwrap_or_else(|| suggestion.to_string())
    };

    let _ = token;
    Some(EditAction {
        start: start_pos - 1,
        end: end_pos - 1,
        label: cleaned,
    })
}

fn apply_edits(
    source_tokens: &[String],
    edits: &[EditAction],
    verb_decode: &HashMap<String, String>,
) -> Vec<String> {
    let mut target_tokens = source_tokens.to_vec();
    let mut shift_idx: isize = 0;

    for edit in edits {
        let target_pos = edit.start + shift_idx;
        if edit.start < 0 {
            continue;
        }

        let source_token = if target_pos >= 0 && (target_pos as usize) < target_tokens.len() {
            target_tokens[target_pos as usize].clone()
        } else {
            String::new()
        };

        if edit.label.is_empty() {
            if target_pos >= 0 && (target_pos as usize) < target_tokens.len() {
                target_tokens.remove(target_pos as usize);
                shift_idx -= 1;
            }
        } else if edit.start == edit.end {
            let word = edit.label.replace("$APPEND_", "");
            let pos = target_pos.max(0) as usize;
            let duplicate_at_pos = target_tokens.get(pos).is_some_and(|token| token == &word);
            let duplicate_before = pos > 0
                && target_tokens
                    .get(pos - 1)
                    .is_some_and(|token| token == &word);
            if duplicate_at_pos || duplicate_before {
                continue;
            }
            target_tokens.insert(pos, word);
            shift_idx += 1;
        } else if edit.label.starts_with("$TRANSFORM_") {
            if target_pos >= 0 && (target_pos as usize) < target_tokens.len() {
                let word = apply_reverse_transformation(&source_token, &edit.label, verb_decode)
                    .unwrap_or(source_token);
                target_tokens[target_pos as usize] = word;
            }
        } else if edit.start == edit.end - 1 {
            if target_pos >= 0 && (target_pos as usize) < target_tokens.len() {
                target_tokens[target_pos as usize] = edit.label.replace("$REPLACE_", "");
            }
        } else if edit.label.starts_with("$MERGE_") {
            let insert_at = (target_pos + 1).max(0) as usize;
            if insert_at <= target_tokens.len() {
                target_tokens.insert(insert_at, edit.label.clone());
                shift_idx += 1;
            }
        }
    }

    replace_merge_transforms(target_tokens)
}

fn replace_merge_transforms(tokens: Vec<String>) -> Vec<String> {
    if tokens.iter().all(|token| !token.starts_with("$MERGE_")) {
        return tokens;
    }

    let mut tokens = tokens;
    if tokens
        .first()
        .is_some_and(|token| token.starts_with("$MERGE_"))
    {
        tokens.remove(0);
    }
    if tokens
        .last()
        .is_some_and(|token| token.starts_with("$MERGE_"))
    {
        tokens.pop();
    }

    let mut line = tokens.join(" ");
    line = line.replace(" $MERGE_HYPHEN ", "-");
    line = line.replace(" $MERGE_SPACE ", "");
    while line.contains("..") || line.contains(",,") || line.contains("??") || line.contains("::") {
        line = line.replace("..", ".");
        line = line.replace(",,", ",");
        line = line.replace("??", "?");
        line = line.replace("::", ":");
    }
    line.split_whitespace().map(ToOwned::to_owned).collect()
}

fn apply_reverse_transformation(
    source_token: &str,
    transform: &str,
    verb_decode: &HashMap<String, String>,
) -> Option<String> {
    if transform == "$KEEP" {
        return Some(source_token.to_string());
    }
    if transform.starts_with("$TRANSFORM_CASE_") {
        return Some(convert_using_case(source_token, transform));
    }
    if let Some(rest) = transform.strip_prefix("$TRANSFORM_VERB_") {
        let key = format!("{source_token}_{rest}");
        return verb_decode.get(&key).cloned();
    }
    if transform.starts_with("$TRANSFORM_SPLIT") {
        return Some(source_token.replace('-', " "));
    }
    if transform.starts_with("$TRANSFORM_AGREEMENT_") {
        return Some(convert_using_plural(source_token, transform));
    }
    None
}

fn convert_using_case(token: &str, action: &str) -> String {
    if action.ends_with("LOWER") {
        token.to_lowercase()
    } else if action.ends_with("UPPER") {
        token.to_uppercase()
    } else if action.ends_with("CAPITAL") {
        capitalize(token)
    } else if action.ends_with("CAPITAL_1") {
        let mut chars = token.chars();
        match chars.next() {
            Some(first) => format!("{}{}", first, capitalize(chars.as_str())),
            None => String::new(),
        }
    } else if action.ends_with("UPPER_-1") {
        let mut chars = token.chars().collect::<Vec<_>>();
        if chars.len() <= 1 {
            token.to_uppercase()
        } else {
            let last = chars.pop().unwrap_or_default();
            format!(
                "{}{}",
                chars.into_iter().collect::<String>().to_uppercase(),
                last
            )
        }
    } else {
        token.to_string()
    }
}

fn convert_using_plural(token: &str, action: &str) -> String {
    if action.ends_with("PLURAL") {
        format!("{token}s")
    } else if action.ends_with("SINGULAR") {
        token
            .chars()
            .take(token.chars().count().saturating_sub(1))
            .collect()
    } else {
        token.to_string()
    }
}

fn capitalize(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str().to_lowercase()),
        None => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Sequence matcher (unchanged from original)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Opcode {
    tag: String,
    i1: usize,
    i2: usize,
    j1: usize,
    j2: usize,
}

fn sequence_matcher(a: &[String], b: &[String]) -> Vec<Opcode> {
    let a_len = a.len();
    let b_len = b.len();
    let mut dp = vec![vec![0_usize; b_len + 1]; a_len + 1];

    for i in (0..a_len).rev() {
        for j in (0..b_len).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i][j].max(dp[i + 1][j])
            };
        }
    }

    let mut i = 0;
    let mut j = 0;
    let mut opcodes = Vec::new();
    while i < a_len || j < b_len {
        if i < a_len && j < b_len && a[i] == b[j] {
            let start_i = i;
            let start_j = j;
            while i < a_len && j < b_len && a[i] == b[j] {
                i += 1;
                j += 1;
            }
            opcodes.push(Opcode {
                tag: "equal".to_string(),
                i1: start_i,
                i2: i,
                j1: start_j,
                j2: j,
            });
        } else if j < b_len && (i == a_len || dp[i][j + 1] >= dp[i + 1][j]) {
            let start_j = j;
            while j < b_len && (i == a_len || dp[i][j + 1] >= dp[i + 1][j]) {
                j += 1;
                if i < a_len && j < b_len && a[i] == b[j] {
                    break;
                }
            }
            opcodes.push(Opcode {
                tag: "insert".to_string(),
                i1: i,
                i2: i,
                j1: start_j,
                j2: j,
            });
        } else if i < a_len {
            let start_i = i;
            while i < a_len && (j == b_len || dp[i + 1][j] > dp[i][j + 1]) {
                i += 1;
                if i < a_len && j < b_len && a[i] == b[j] {
                    break;
                }
            }
            opcodes.push(Opcode {
                tag: "delete".to_string(),
                i1: start_i,
                i2: i,
                j1: j,
                j2: j,
            });
        }
    }
    opcodes
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn read_lines(path: &Path) -> Result<Vec<String>> {
    Ok(fs::read_to_string(path)
        .with_context(|| format!("failed reading {}", path.display()))?
        .lines()
        .map(ToOwned::to_owned)
        .collect())
}

fn read_verb_decode_map(path: &Path) -> Result<HashMap<String, String>> {
    let mut decode = HashMap::new();
    for line in read_lines(path)? {
        let Some((words, tags)) = line.split_once(':') else {
            continue;
        };
        let Some((word1, word2)) = words.split_once('_') else {
            continue;
        };
        let Some((tag1, tag2)) = tags.split_once('_') else {
            continue;
        };
        let key = format!("{word1}_{tag1}_{}", tag2.trim());
        decode.entry(key).or_insert_with(|| word2.to_string());
    }
    Ok(decode)
}

fn python_index(len: usize, index: isize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    if index >= 0 {
        let index = index as usize;
        return (index < len).then_some(index);
    }
    let index = len as isize + index;
    (index >= 0).then_some(index as usize)
}

fn softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = values
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let sum = exps.iter().sum::<f32>();
    exps.into_iter().map(|value| value / sum).collect()
}

fn argmax(values: &[f32]) -> (usize, f32) {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .unwrap_or((0, 0.0))
}

fn strip_spaces_before_punctuation(text: &str, punctuation: &[String]) -> String {
    let mut output = text.to_string();
    for punct in punctuation {
        let with_space = format!(" {punct}");
        output = output.replace(&with_space, punct);
    }
    output
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

// ---------------------------------------------------------------------------
// Report types: snippet benchmarking (unchanged)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct CapuSnippet {
    id: String,
    raw: String,
    clean_lower: String,
    capu: String,
}

#[derive(Debug, Clone, Serialize)]
struct ParityResult {
    id: String,
    expected: String,
    python: String,
    rust_onnx: String,
    python_matches_expected: bool,
    rust_matches_expected: bool,
    python_matches_rust: bool,
}

#[derive(Debug, Clone, Serialize)]
struct LatencySummary {
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct EngineBenchmark {
    init_ms: f64,
    first_inference_ms: f64,
    warmup_iterations: usize,
    sample_text: String,
    sample_output: String,
    warm_summary: LatencySummary,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkReport {
    export_dir: String,
    snippets_path: String,
    ort_tuning: OrtTuning,
    benchmark_id: String,
    parity: Vec<ParityResult>,
    python_worker: EngineBenchmark,
    rust_onnx: EngineBenchmark,
    recommendation: String,
}

// ---------------------------------------------------------------------------
// Report types: audio-derived benchmark (req #4, #5, #6)
// ---------------------------------------------------------------------------

/// Per-case timing summary for the Rust ONNX pipeline.
#[derive(Debug, Clone, Serialize)]
struct RustOnnxBenchmark {
    init_ms: f64,
    first_inference_ms: f64,
    warmup_iterations: usize,
    sample_text: String,
    sample_output: String,
    warm_summary: LatencySummary,
    /// Per-stage mean / p50 / p95 across warm iterations.
    stage_timings: StageTimingsSummary,
}

/// Aggregated stage-level timing percentiles.
#[derive(Debug, Clone, Serialize)]
struct StageTimingsSummary {
    mean: StageTimings,
    p50: StageTimings,
    p95: StageTimings,
}

/// One benchmark case (one prefix length of one asset).
#[derive(Debug, Clone, Serialize)]
struct BenchmarkCase {
    label: String,
    char_count: usize,
    /// Estimated audio duration for this prefix (proportional to character ratio).
    estimated_audio_seconds: f64,
    estimated: bool,
    iterations: usize,
    /// Python CAPU worker results.
    python: EngineBenchmark,
    /// Rust ONNX results with stage detail.
    rust: RustOnnxBenchmark,
    /// slowdown_ratio = rust mean / python mean.
    slowdown_ratio: f64,
    /// Whether the final Rust output exactly matches the Python output for this case.
    output_parity: bool,
}

/// One audio asset with all its prefix-length benchmark cases.
#[derive(Debug, Clone, Serialize)]
struct AssetBenchmark {
    asset: String,
    total_chars: usize,
    total_audio_seconds: f64,
    cases: Vec<BenchmarkCase>,
}

/// Top-level report for the audio-derived benchmark.
#[derive(Debug, Clone, Serialize)]
struct AssetsBenchmarkReport {
    export_dir: String,
    ort_tuning: OrtTuning,
    model_id: String,
    assets: Vec<AssetBenchmark>,
    recommendation: String,
}

// ---------------------------------------------------------------------------
// Helpers for snippet-based benchmark
// ---------------------------------------------------------------------------

fn read_snippets(path: &Path) -> Result<Vec<CapuSnippet>> {
    fs::read_to_string(path)
        .with_context(|| format!("failed reading snippets: {}", path.display()))?
        .lines()
        .map(|line| serde_json::from_str::<CapuSnippet>(line).map_err(Into::into))
        .collect()
}

fn latency_summary(samples: &[f64]) -> LatencySummary {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mean_ms = if sorted.is_empty() {
        0.0
    } else {
        sorted.iter().sum::<f64>() / sorted.len() as f64
    };
    let percentile = |fraction: f64| -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
        sorted[index]
    };
    LatencySummary {
        mean_ms,
        p50_ms: percentile(0.50),
        p95_ms: percentile(0.95),
        min_ms: *sorted.first().unwrap_or(&0.0),
        max_ms: *sorted.last().unwrap_or(&0.0),
    }
}

fn stage_summary(samples: &[StageTimings]) -> StageTimingsSummary {
    let n = samples.len();
    if n == 0 {
        return StageTimingsSummary {
            mean: StageTimings::default(),
            p50: StageTimings::default(),
            p95: StageTimings::default(),
        };
    }

    let mean =
        |f: fn(&StageTimings) -> f64| -> f64 { samples.iter().map(f).sum::<f64>() / n as f64 };
    let percentile = |f: fn(&StageTimings) -> f64, pct: f64| -> f64 {
        let mut vals: Vec<f64> = samples.iter().map(f).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let idx = ((vals.len() - 1) as f64 * pct).round() as usize;
        vals[idx]
    };

    let apply = |f: fn(&StageTimings) -> f64| -> (f64, f64, f64) {
        (mean(f), percentile(f, 0.50), percentile(f, 0.95))
    };

    let (cm, c50, c95) = apply(|s| s.clean_ms);
    let (sm, s50, s95) = apply(|s| s.split_chunk_ms);
    let (tm, t50, t95) = apply(|s| s.tokenize_ms);
    let (om, o50, o95) = apply(|s| s.ort_ms);
    let (som, so50, so95) = apply(|s| s.softmax_ms);
    let (em, e50, e95) = apply(|s| s.edit_apply_ms);
    let (mm, m50, m95) = apply(|s| s.merge_ms);
    let (fm, f50, f95) = apply(|s| s.finalize_ms);
    let (tolm, tol50, tol95) = apply(|s| s.total_ms);

    StageTimingsSummary {
        mean: StageTimings {
            clean_ms: cm,
            split_chunk_ms: sm,
            tokenize_ms: tm,
            ort_ms: om,
            softmax_ms: som,
            edit_apply_ms: em,
            merge_ms: mm,
            finalize_ms: fm,
            total_ms: tolm,
        },
        p50: StageTimings {
            clean_ms: c50,
            split_chunk_ms: s50,
            tokenize_ms: t50,
            ort_ms: o50,
            softmax_ms: so50,
            edit_apply_ms: e50,
            merge_ms: m50,
            finalize_ms: f50,
            total_ms: tol50,
        },
        p95: StageTimings {
            clean_ms: c95,
            split_chunk_ms: s95,
            tokenize_ms: t95,
            ort_ms: o95,
            softmax_ms: so95,
            edit_apply_ms: e95,
            merge_ms: m95,
            finalize_ms: f95,
            total_ms: tol95,
        },
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();
    let workspace = WorkspacePaths::new(&cli.workspace_root);
    let export_dir = workspace.resolve(&cli.export_dir);
    let ort_tuning = OrtTuning {
        intra_threads: cli.ort_intra_threads,
        memory_pattern: cli.ort_memory_pattern,
    };

    match cli.command {
        Command::Run { text } => {
            let start = Instant::now();
            let mut runner = CapuOnnxRunner::load(&export_dir, ort_tuning)?;
            let init_ms = ms(start);
            let started = Instant::now();
            let output = runner.process_text(&text)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "text": text,
                    "output": output,
                    "init_ms": init_ms,
                    "inference_ms": ms(started),
                    "ort_tuning": ort_tuning,
                }))?
            );
        }

        // -----------------------------------------------------------------------
        // Existing snippet-based benchmark (preserved)
        // -----------------------------------------------------------------------
        Command::Benchmark {
            snippets,
            benchmark_id,
            iterations,
            output,
        } => {
            let snippets_path = workspace.resolve(&snippets);
            let snippets = read_snippets(&snippets_path)?;
            let benchmark_snippet = snippets
                .iter()
                .find(|snippet| snippet.id == benchmark_id)
                .cloned()
                .with_context(|| format!("benchmark snippet not found: {benchmark_id}"))?;

            let runtime = workspace.load_runtime_config(None)?;
            let registry = workspace.load_registry(&runtime)?;
            let capu_model = registry
                .capu_models
                .first()
                .context("CAPU model registry is empty")?;

            let python_init = Instant::now();
            let (_, python_postprocessor) =
                build_capu_postprocessor(&workspace, &runtime.capu, capu_model)?;
            let python_init_ms = ms(python_init);

            let rust_init = Instant::now();
            let mut runner = CapuOnnxRunner::load(&export_dir, ort_tuning)?;
            let rust_init_ms = ms(rust_init);

            let mut parity = Vec::with_capacity(snippets.len());
            for snippet in &snippets {
                let python = python_postprocessor.process_text(&snippet.raw)?;
                let rust_onnx = runner.process_text(&snippet.raw)?;
                parity.push(ParityResult {
                    id: snippet.id.clone(),
                    expected: snippet.capu.clone(),
                    python_matches_expected: python == snippet.capu,
                    rust_matches_expected: rust_onnx == snippet.capu,
                    python_matches_rust: python == rust_onnx,
                    python,
                    rust_onnx,
                });
            }

            if parity
                .iter()
                .any(|item| !item.python_matches_rust || !item.rust_matches_expected)
            {
                bail!("parity check failed; inspect JSON output for details");
            }

            let python_first = Instant::now();
            let python_output =
                python_postprocessor.process_text(&benchmark_snippet.clean_lower)?;
            let python_first_ms = ms(python_first);
            let mut python_samples = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let start = Instant::now();
                let _ = python_postprocessor.process_text(&benchmark_snippet.clean_lower)?;
                python_samples.push(ms(start));
            }

            let rust_first = Instant::now();
            let rust_output = runner.process_text(&benchmark_snippet.clean_lower)?;
            let rust_first_ms = ms(rust_first);
            let mut rust_samples = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let start = Instant::now();
                let _ = runner.process_text(&benchmark_snippet.clean_lower)?;
                rust_samples.push(ms(start));
            }

            let report = BenchmarkReport {
                export_dir: export_dir.display().to_string(),
                snippets_path: snippets_path.display().to_string(),
                ort_tuning,
                benchmark_id: benchmark_id.clone(),
                parity,
                python_worker: EngineBenchmark {
                    init_ms: python_init_ms,
                    first_inference_ms: python_first_ms,
                    warmup_iterations: iterations,
                    sample_text: benchmark_snippet.clean_lower.clone(),
                    sample_output: python_output,
                    warm_summary: latency_summary(&python_samples),
                },
                rust_onnx: EngineBenchmark {
                    init_ms: rust_init_ms,
                    first_inference_ms: rust_first_ms,
                    warmup_iterations: iterations,
                    sample_text: benchmark_snippet.clean_lower.clone(),
                    sample_output: rust_output,
                    warm_summary: latency_summary(&rust_samples),
                },
                recommendation: "Use the ONNX + Rust path as a validated spike only after export artifacts are present; keep the Python worker as the default production CAPU path until broader fixture coverage and packaging are added.".to_string(),
            };

            let report_json = serde_json::to_string_pretty(&report)?;
            if let Some(output_path) = output {
                let output_path = workspace.resolve(&output_path);
                if let Some(parent) = output_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&output_path, &report_json).with_context(|| {
                    format!("failed writing report to {}", output_path.display())
                })?;
            }
            println!("{report_json}");
        }

        // -----------------------------------------------------------------------
        // Audio-derived asset benchmark (req #4, #5, #6)
        // -----------------------------------------------------------------------
        Command::BenchmarkAssets {
            model_id,
            assets_dir,
            iterations,
            warmup,
            output,
        } => {
            let assets_dir = workspace.resolve(&assets_dir);
            let model_id = if model_id.is_empty() {
                "vit_stt_vi_v2_capu".to_string()
            } else {
                model_id
            };

            // Files to benchmark (req #4)
            let asset_files = [
                ("weanxinviec.mp3", "weanxinviec"),
                ("thoitiet936.mp3", "thoitiet936"),
                ("bomman.mp3", "bomman"),
            ];

            // Target lengths to probe (req #4)
            let target_labels = &[
                ("30s", 30.0),
                ("2m", 120.0),
                ("4m", 240.0),
                ("8m", 480.0),
                ("12m", 720.0),
                ("full", f64::INFINITY),
            ];

            // Init SttRuntime
            let stt = SttRuntime::load(Some(cli.workspace_root.clone()), None)?;

            // Init Python CAPU worker
            let runtime = workspace.load_runtime_config(None)?;
            let capu_model = stt.resolve_capu_model(None)?;
            let python_init = Instant::now();
            let (_, python_postprocessor) =
                build_capu_postprocessor(&workspace, &runtime.capu, capu_model)?;
            let python_init_ms = ms(python_init);

            // Init Rust ONNX runner
            let rust_init = Instant::now();
            let mut runner = CapuOnnxRunner::load(&export_dir, ort_tuning)?;
            let rust_init_ms = ms(rust_init);

            let cleaner = BuiltinPostprocessor::new(PostprocessMode::CleanLower);

            let mut asset_results: Vec<AssetBenchmark> = Vec::with_capacity(asset_files.len());

            for (filename, asset_name) in &asset_files {
                let asset_path = assets_dir.join(filename);
                eprintln!("[capu_onnx_spike] transcribing asset: {filename}");

                // Transcribe the full audio to obtain the clean_lower transcript
                let result = stt
                    .transcribe_path(&model_id, &asset_path)
                    .with_context(|| format!("failed transcribing {filename}"))?;
                let total_audio_seconds = result.duration as f64;

                let full_text = result.text.trim().to_string();
                let clean_lower_text = cleaner.process_text(&full_text)?;
                let total_chars = clean_lower_text.chars().count();
                let words: Vec<&str> = clean_lower_text.split_whitespace().collect();

                // Build prefix-based benchmark cases
                let mut cases: Vec<BenchmarkCase> = Vec::with_capacity(target_labels.len());
                let mut seen_char_counts: HashSet<usize> = HashSet::new();

                for &(label, target_seconds) in target_labels {
                    // Build prefix text
                    let prefix_text: String =
                        if target_seconds.is_infinite() || target_seconds >= total_audio_seconds {
                            clean_lower_text.clone()
                        } else {
                            let ratio = (target_seconds / total_audio_seconds).min(1.0);
                            let target_chars = (total_chars as f64 * ratio) as usize;
                            let mut acc = 0usize;
                            let mut n_words = 0usize;
                            for w in &words {
                                let next = acc + w.chars().count() + 1;
                                if next > target_chars && n_words > 0 {
                                    break;
                                }
                                acc = next;
                                n_words += 1;
                            }
                            words[..n_words.max(1)].join(" ")
                        };

                    if prefix_text.is_empty() {
                        continue;
                    }

                    eprintln!(
                        "[capu_onnx_spike] benchmarking case: {asset_name}/{label} (chars={})",
                        prefix_text.chars().count()
                    );

                    let prefix_chars = prefix_text.chars().count();
                    if !seen_char_counts.insert(prefix_chars) {
                        continue;
                    }
                    let estimated_seconds =
                        total_audio_seconds * (prefix_chars as f64 / total_chars as f64);
                    let is_estimated = true; // always estimated from character ratio

                    // === Python worker benchmark ===
                    let python_first = Instant::now();
                    let python_output = python_postprocessor.process_text(&prefix_text)?;
                    let python_first_ms = ms(python_first);

                    let mut python_samples = Vec::with_capacity(iterations);
                    for run_index in 0..(iterations + warmup) {
                        let s = Instant::now();
                        let _ = python_postprocessor.process_text(&prefix_text)?;
                        let elapsed = ms(s);
                        if run_index >= warmup {
                            python_samples.push(elapsed);
                        }
                    }

                    let python_bench = EngineBenchmark {
                        init_ms: python_init_ms,
                        first_inference_ms: python_first_ms,
                        warmup_iterations: warmup,
                        sample_text: prefix_text.clone(),
                        sample_output: python_output.clone(),
                        warm_summary: latency_summary(&python_samples),
                    };

                    // === Rust ONNX benchmark ===
                    // Warmup first
                    let rust_first = Instant::now();
                    let (rust_output, _first_timings) =
                        runner.process_text_with_timings(&prefix_text)?;
                    let rust_first_ms = ms(rust_first);

                    let mut rust_latencies = Vec::with_capacity(iterations);
                    let mut rust_stages = Vec::with_capacity(iterations);
                    for run_index in 0..(iterations + warmup) {
                        let s = Instant::now();
                        let (_, stage) = runner.process_text_with_timings(&prefix_text)?;
                        let elapsed = ms(s);
                        if run_index >= warmup {
                            rust_latencies.push(elapsed);
                            rust_stages.push(stage);
                        }
                    }

                    let rust_bench = RustOnnxBenchmark {
                        init_ms: rust_init_ms,
                        first_inference_ms: rust_first_ms,
                        warmup_iterations: warmup,
                        sample_text: prefix_text.clone(),
                        sample_output: rust_output.clone(),
                        warm_summary: latency_summary(&rust_latencies),
                        stage_timings: stage_summary(&rust_stages),
                    };

                    let python_mean = python_bench.warm_summary.mean_ms;
                    let rust_mean = rust_bench.warm_summary.mean_ms;
                    let slowdown_ratio = if python_mean > 0.0 {
                        rust_mean / python_mean
                    } else {
                        f64::NAN
                    };

                    cases.push(BenchmarkCase {
                        label: format!("{asset_name}/{label}"),
                        char_count: prefix_chars,
                        estimated_audio_seconds: estimated_seconds,
                        estimated: is_estimated,
                        iterations,
                        python: python_bench,
                        rust: rust_bench,
                        slowdown_ratio,
                        output_parity: python_output == rust_output,
                    });
                }

                asset_results.push(AssetBenchmark {
                    asset: asset_name.to_string(),
                    total_chars,
                    total_audio_seconds,
                    cases,
                });
            }

            let report = AssetsBenchmarkReport {
                export_dir: export_dir.display().to_string(),
                ort_tuning,
                model_id: model_id.to_string(),
                assets: asset_results,
                recommendation: format!(
                    "Benchmark across {} audio assets with {} iterations per case. \
                     Compare Rust ONNX vs Python CAPU slowdown ratios and stage breakdown \
                     to identify optimization targets.",
                    asset_files.len(),
                    iterations,
                ),
            };

            let report_json = serde_json::to_string_pretty(&report)?;
            if let Some(output_path) = output {
                let output_path = workspace.resolve(&output_path);
                if let Some(parent) = output_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&output_path, &report_json).with_context(|| {
                    format!("failed writing report to {}", output_path.display())
                })?;
            }
            println!("{report_json}");
        }
    }

    Ok(())
}
