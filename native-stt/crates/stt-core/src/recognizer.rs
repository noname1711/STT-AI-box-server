use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig, TenVadModelConfig,
    VadModelConfig, VoiceActivityDetector,
};
use tracing::info;

use crate::assets::resolve_offline_model_assets;
use crate::config::ResolvedModelConfig;
use crate::types::{RecognizedSegment, TranscriptionResult};

const MIN_DECODER_AUDIO_MS: usize = 1_000;

pub struct RecognizerRuntime {
    model: ResolvedModelConfig,
}

pub struct WarmedRecognizerRuntime {
    model: ResolvedModelConfig,
    recognizer: OfflineRecognizer,
}

impl std::fmt::Debug for WarmedRecognizerRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WarmedRecognizerRuntime")
            .field("model_id", &self.model.id)
            .finish_non_exhaustive()
    }
}

impl RecognizerRuntime {
    pub fn new(model: ResolvedModelConfig) -> Self {
        Self { model }
    }

    fn build_recognizer(&self) -> Result<OfflineRecognizer> {
        let assets = resolve_offline_model_assets(
            &self.model.model_dir,
            &self.model.model_type,
            self.model.encoder.as_deref(),
            self.model.decoder.as_deref(),
            self.model.joiner.as_deref(),
            self.model.tokens.as_deref(),
            self.model.bpe_vocab.as_deref(),
        )?;

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.transducer = OfflineTransducerModelConfig {
            encoder: Some(assets.encoder.display().to_string()),
            decoder: Some(assets.decoder.display().to_string()),
            joiner: Some(assets.joiner.display().to_string()),
        };
        config.model_config.tokens = Some(assets.tokens.display().to_string());
        config.model_config.provider = Some(self.model.provider.clone());
        config.model_config.num_threads = self.model.num_threads;
        config.model_config.model_type = Some(assets.model_type.clone());
        // modeling_unit/bpe_vocab are needed only when an explicit sherpa
        // contextual-bias BPE vocabulary is configured. Normal transducer
        // decoding uses encoder/decoder/joiner + tokens and must not parse
        // SentencePiece bpe.model as a text vocabulary.
        if let Some(bpe) = assets.bpe_vocab.as_ref() {
            config.model_config.modeling_unit = Some("bpe".to_string());
            config.model_config.bpe_vocab = Some(bpe.display().to_string());
        }
        // HL Meet v17 PASS 2: search multiple transducer hypotheses for FINAL.
        config.decoding_method = Some("modified_beam_search".to_string());
        if self.model.id == "vit_stt_vi_v2" {
            config.max_active_paths = 64;
            config.blank_penalty = 2.0;
        } else {
            // Preserve the Phase6B decoder baseline for every un-tuned model,
            // including the English Parakeet model.
            config.max_active_paths = 16;
            config.blank_penalty = 0.0;
        }

        OfflineRecognizer::create(&config)
            .ok_or_else(|| anyhow!("failed to create offline recognizer for {}", self.model.id))
    }

    fn build_vad(&self) -> Result<VoiceActivityDetector> {
        build_vad_for_model(&self.model)
    }

    fn decode_segment(
        recognizer: &OfflineRecognizer,
        samples: &[f32],
        sample_rate: i32,
    ) -> Result<String> {
        if samples.is_empty() || sample_rate <= 0 {
            return Ok(String::new());
        }
        let prepared = pad_for_decoder(samples, sample_rate);
        let stream = recognizer.create_stream();
        stream.accept_waveform(sample_rate, &prepared);
        recognizer.decode(&stream);
        Ok(stream
            .get_result()
            .map(|result| result.text.trim().to_string())
            .unwrap_or_default())
    }

    fn segment_with_vad(
        &self,
        samples: &[f32],
        sample_rate: i32,
    ) -> Result<Vec<(f32, f32, Vec<f32>)>> {
        let vad = self.build_vad()?;
        segment_with_vad(vad, samples, sample_rate)
    }

    pub fn transcribe_samples(
        &self,
        samples: &[f32],
        sample_rate: i32,
    ) -> Result<TranscriptionResult> {
        let recognizer_started = Instant::now();
        info!(
            model_id = self.model.id.as_str(),
            provider = self.model.provider.as_str(),
            num_threads = self.model.num_threads,
            "initializing offline recognizer"
        );
        let recognizer = self.build_recognizer()?;
        info!(
            model_id = self.model.id.as_str(),
            elapsed_ms = recognizer_started.elapsed().as_millis(),
            "offline recognizer ready"
        );
        let started = Instant::now();
        let duration = samples.len() as f32 / sample_rate as f32;

        let segments = if self.model.vad.enabled {
            self.segment_with_vad(samples, sample_rate)?
        } else {
            Vec::new()
        };

        info!(
            model_id = self.model.id.as_str(),
            vad_enabled = self.model.vad.enabled,
            segment_count = segments.len(),
            duration_seconds = duration,
            "segmentation prepared"
        );

        let result = if self.model.vad.enabled && segments.is_empty() {
            empty_transcription_result(&self.model, duration, started)
        } else if segments.is_empty() {
            let text = Self::decode_segment(&recognizer, samples, sample_rate)?;
            TranscriptionResult {
                text,
                model: self.model.id.clone(),
                language: self.model.language.clone(),
                duration,
                processing_time: started.elapsed().as_secs_f32(),
                segments: None,
            }
        } else {
            let mut output_segments = Vec::new();
            let mut texts = Vec::new();
            for (start, end, seg_samples) in segments {
                let text = Self::decode_segment(&recognizer, &seg_samples, sample_rate)?;
                if !text.trim().is_empty() {
                    texts.push(text.clone());
                    output_segments.push(RecognizedSegment { start, end, text });
                }
            }

            TranscriptionResult {
                text: texts.join(" "),
                model: self.model.id.clone(),
                language: self.model.language.clone(),
                duration,
                processing_time: started.elapsed().as_secs_f32(),
                segments: if output_segments.is_empty() {
                    None
                } else {
                    Some(output_segments)
                },
            }
        };

        info!(
            model_id = self.model.id.as_str(),
            processing_seconds = result.processing_time,
            segment_count = result.segments.as_ref().map_or(0, Vec::len),
            text_chars = result.text.chars().count(),
            "recognition finished"
        );

        Ok(result)
    }

    pub fn transcribe_probe(&self) -> Result<TranscriptionResult> {
        let path = self
            .model
            .startup_probe_wav_path
            .as_ref()
            .context("startup probe wav path is not configured")?;
        let decoded = crate::audio::decode_audio_path(path)?;
        self.transcribe_samples(&decoded.samples, decoded.sample_rate)
    }

    pub fn warm(self) -> Result<WarmedRecognizerRuntime> {
        WarmedRecognizerRuntime::new(self.model)
    }
}

impl WarmedRecognizerRuntime {
    pub fn new(model: ResolvedModelConfig) -> Result<Self> {
        let recognizer_started = Instant::now();
        info!(
            model_id = model.id.as_str(),
            provider = model.provider.as_str(),
            num_threads = model.num_threads,
            "initializing offline recognizer"
        );
        let recognizer = RecognizerRuntime::new(model.clone()).build_recognizer()?;
        info!(
            model_id = model.id.as_str(),
            elapsed_ms = recognizer_started.elapsed().as_millis(),
            "offline recognizer ready"
        );
        Ok(Self { model, recognizer })
    }

    pub fn model(&self) -> &ResolvedModelConfig {
        &self.model
    }

    pub fn transcribe_samples(
        &self,
        samples: &[f32],
        sample_rate: i32,
    ) -> Result<TranscriptionResult> {
        let started = Instant::now();
        let duration = samples.len() as f32 / sample_rate as f32;

        let segments = if self.model.vad.enabled {
            segment_with_vad(build_vad_for_model(&self.model)?, samples, sample_rate)?
        } else {
            Vec::new()
        };

        info!(
            model_id = self.model.id.as_str(),
            vad_enabled = self.model.vad.enabled,
            segment_count = segments.len(),
            duration_seconds = duration,
            "segmentation prepared"
        );

        let result = if self.model.vad.enabled && segments.is_empty() {
            empty_transcription_result(&self.model, duration, started)
        } else if segments.is_empty() {
            let text = RecognizerRuntime::decode_segment(&self.recognizer, samples, sample_rate)?;
            TranscriptionResult {
                text,
                model: self.model.id.clone(),
                language: self.model.language.clone(),
                duration,
                processing_time: started.elapsed().as_secs_f32(),
                segments: None,
            }
        } else {
            let mut output_segments = Vec::new();
            let mut texts = Vec::new();
            for (start, end, seg_samples) in segments {
                let text =
                    RecognizerRuntime::decode_segment(&self.recognizer, &seg_samples, sample_rate)?;
                if !text.trim().is_empty() {
                    texts.push(text.clone());
                    output_segments.push(RecognizedSegment { start, end, text });
                }
            }

            TranscriptionResult {
                text: texts.join(" "),
                model: self.model.id.clone(),
                language: self.model.language.clone(),
                duration,
                processing_time: started.elapsed().as_secs_f32(),
                segments: if output_segments.is_empty() {
                    None
                } else {
                    Some(output_segments)
                },
            }
        };

        info!(
            model_id = self.model.id.as_str(),
            processing_seconds = result.processing_time,
            segment_count = result.segments.as_ref().map_or(0, Vec::len),
            text_chars = result.text.chars().count(),
            "recognition finished"
        );

        Ok(result)
    }
}

fn pad_for_decoder(samples: &[f32], sample_rate: i32) -> Vec<f32> {
    let minimum_samples = (sample_rate.max(1) as usize)
        .saturating_mul(MIN_DECODER_AUDIO_MS)
        .saturating_div(1_000);
    if samples.len() >= minimum_samples {
        return samples.to_vec();
    }

    let missing = minimum_samples.saturating_sub(samples.len());
    let leading = missing / 2;
    let trailing = missing.saturating_sub(leading);
    let mut padded = Vec::with_capacity(minimum_samples);
    padded.resize(leading, 0.0);
    padded.extend_from_slice(samples);
    padded.resize(padded.len().saturating_add(trailing), 0.0);
    padded
}

fn empty_transcription_result(
    model: &ResolvedModelConfig,
    duration: f32,
    started: Instant,
) -> TranscriptionResult {
    TranscriptionResult {
        text: String::new(),
        model: model.id.clone(),
        language: model.language.clone(),
        duration,
        processing_time: started.elapsed().as_secs_f32(),
        segments: None,
    }
}

fn build_vad_for_model(model: &ResolvedModelConfig) -> Result<VoiceActivityDetector> {
    let ten = TenVadModelConfig {
        model: Some(model.vad_model_path.display().to_string()),
        threshold: model.vad.threshold,
        min_silence_duration: model.vad.min_silence,
        min_speech_duration: model.vad.min_speech,
        max_speech_duration: model.vad.max_speech,
        window_size: 256,
    };

    let config = VadModelConfig {
        silero_vad: Default::default(),
        ten_vad: ten,
        sample_rate: 16_000,
        num_threads: 1,
        provider: Some("cpu".to_string()),
        debug: false,
    };

    VoiceActivityDetector::create(&config, 100.0).ok_or_else(|| anyhow!("failed to create VAD"))
}

fn segment_with_vad(
    vad: VoiceActivityDetector,
    samples: &[f32],
    sample_rate: i32,
) -> Result<Vec<(f32, f32, Vec<f32>)>> {
    let window_size = 512usize;
    let pad_samples = ((0.5 * sample_rate as f32) as usize / window_size) * window_size;
    let mut padded = vec![0.0f32; pad_samples];
    padded.extend_from_slice(samples);
    let pad_duration = pad_samples as f32 / sample_rate as f32;

    for chunk in padded.chunks(window_size) {
        if chunk.len() == window_size {
            vad.accept_waveform(chunk);
        }
    }

    vad.flush();

    let mut segments = Vec::new();
    while vad.front().is_some() {
        let (start, seg_samples) = {
            let segment = vad.front().context("VAD front unexpectedly missing")?;
            let raw_start = segment.start() as f32 / sample_rate as f32;
            let start = (raw_start - pad_duration).max(0.0);
            let seg_samples = segment.samples().to_vec();
            (start, seg_samples)
        };
        let end = start + seg_samples.len() as f32 / sample_rate as f32;
        segments.push((start, end, seg_samples));
        vad.pop();
    }

    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_decoder_input_is_center_padded_to_one_second() {
        let samples = vec![0.25; 1_600];
        let padded = pad_for_decoder(&samples, 16_000);

        assert_eq!(padded.len(), 16_000);
        assert!(padded[..7_200].iter().all(|sample| *sample == 0.0));
        assert_eq!(&padded[7_200..8_800], samples.as_slice());
        assert!(padded[8_800..].iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn decoder_input_at_safe_length_is_unchanged() {
        let samples = vec![0.25; 16_000];
        assert_eq!(pad_for_decoder(&samples, 16_000), samples);
    }
}
