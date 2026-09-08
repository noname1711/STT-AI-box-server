use serde::{Deserialize, Serialize};

use crate::postprocess::PostprocessMode;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecognizedSegment {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionResult {
    pub text: String,
    pub model: String,
    pub language: String,
    pub duration: f32,
    pub processing_time: f32,
    pub segments: Option<Vec<RecognizedSegment>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelSummary {
    pub id: String,
    pub language: String,
    pub postprocess_mode: PostprocessMode,
}
