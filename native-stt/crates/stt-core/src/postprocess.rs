use anyhow::Result;
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::types::{RecognizedSegment, TranscriptionResult};

const REMOVED_CHARS: [char; 5] = ['\u{feff}', '\u{200b}', '\u{200c}', '\u{200d}', '\u{2060}'];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PostprocessMode {
    CleanLower,
    Normalize,
    Capu,
    None,
}

impl PostprocessMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CleanLower => "clean_lower",
            Self::Normalize => "normalize",
            Self::Capu => "capu",
            Self::None => "none",
        }
    }
}

impl std::fmt::Display for PostprocessMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

pub fn normalize_text(text: &str, lowercase: bool) -> String {
    let mut cleaned = String::new();

    for ch in text.nfc() {
        if REMOVED_CHARS.contains(&ch) {
            continue;
        }
        if ch.is_whitespace() {
            cleaned.push(' ');
            continue;
        }
        if ch.is_control() {
            continue;
        }
        cleaned.push(ch);
    }

    let collapsed = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    if lowercase {
        collapsed.to_lowercase()
    } else {
        collapsed
    }
}

pub fn clean_text(text: &str) -> String {
    normalize_text(text, true)
}

pub fn finalize_text(text: &str) -> String {
    normalize_text(text, false)
}

pub trait Postprocessor: Send + Sync {
    fn mode(&self) -> PostprocessMode;

    fn process_text(&self, text: &str) -> Result<String>;

    fn combine_segment_texts(&self, texts: &[String]) -> Result<String> {
        Ok(texts.join(" "))
    }

    fn process_result(&self, result: &TranscriptionResult) -> Result<TranscriptionResult> {
        if let Some(segments) = &result.segments {
            let mut processed_segments = Vec::with_capacity(segments.len());
            let mut texts = Vec::with_capacity(segments.len());
            for segment in segments {
                let text = self.process_text(&segment.text)?;
                texts.push(text.clone());
                processed_segments.push(RecognizedSegment {
                    start: segment.start,
                    end: segment.end,
                    text,
                });
            }

            return Ok(TranscriptionResult {
                text: self.combine_segment_texts(&texts)?,
                model: result.model.clone(),
                language: result.language.clone(),
                duration: result.duration,
                processing_time: result.processing_time,
                segments: Some(processed_segments),
            });
        }

        Ok(TranscriptionResult {
            text: self.process_text(&result.text)?,
            model: result.model.clone(),
            language: result.language.clone(),
            duration: result.duration,
            processing_time: result.processing_time,
            segments: None,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BuiltinPostprocessor {
    mode: PostprocessMode,
}

impl BuiltinPostprocessor {
    pub fn new(mode: PostprocessMode) -> Self {
        Self { mode }
    }
}

impl Default for BuiltinPostprocessor {
    fn default() -> Self {
        Self {
            mode: PostprocessMode::None,
        }
    }
}

impl Postprocessor for BuiltinPostprocessor {
    fn mode(&self) -> PostprocessMode {
        self.mode
    }

    fn process_text(&self, text: &str) -> Result<String> {
        Ok(match self.mode {
            PostprocessMode::CleanLower => clean_text(text),
            PostprocessMode::Normalize => finalize_text(text),
            PostprocessMode::None => text.to_string(),
            PostprocessMode::Capu => clean_text(text),
        })
    }

    fn combine_segment_texts(&self, texts: &[String]) -> Result<String> {
        Ok(match self.mode {
            PostprocessMode::CleanLower => clean_text(&texts.join(" ")),
            PostprocessMode::Normalize => finalize_text(&texts.join(" ")),
            PostprocessMode::None => texts.join(" "),
            PostprocessMode::Capu => clean_text(&texts.join(" ")),
        })
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NormalizeTextPostprocessor;

impl Postprocessor for NormalizeTextPostprocessor {
    fn mode(&self) -> PostprocessMode {
        PostprocessMode::Normalize
    }

    fn process_text(&self, text: &str) -> Result<String> {
        Ok(finalize_text(text))
    }

    fn combine_segment_texts(&self, texts: &[String]) -> Result<String> {
        Ok(finalize_text(&texts.join(" ")))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoopPostprocessor;

impl Postprocessor for NoopPostprocessor {
    fn mode(&self) -> PostprocessMode {
        PostprocessMode::None
    }

    fn process_text(&self, text: &str) -> Result<String> {
        Ok(text.to_string())
    }

    fn process_result(&self, result: &TranscriptionResult) -> Result<TranscriptionResult> {
        Ok(result.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_hidden_chars_and_whitespace() {
        let text = " Xin\u{200b}  chào\n\tViệt Nam\u{feff} ";
        assert_eq!(clean_text(text), "xin chào việt nam");
        assert_eq!(finalize_text(text), "Xin chào Việt Nam");
    }
}
