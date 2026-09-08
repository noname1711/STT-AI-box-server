use std::path::PathBuf;

use anyhow::Result;

use stt_capu::{build_capu_postprocessor, probe_onnx_or_pure_rust_support};
use stt_core::config::WorkspacePaths;
use stt_core::postprocess::Postprocessor;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

#[test]
fn capu_worker_matches_locked_output() -> Result<()> {
    let root = workspace_root();
    let workspace = WorkspacePaths::new(&root);
    let runtime = workspace.load_runtime_config(None)?;
    let registry = workspace.load_registry(&runtime)?;
    let capu_model = registry
        .capu_models
        .first()
        .expect("CAPU model registry should not be empty");

    let spike = probe_onnx_or_pure_rust_support(&workspace, capu_model);
    assert!(!spike.supported);
    assert!(spike.reason.contains("PyTorch-only"));

    let (_, capu) = build_capu_postprocessor(&workspace, &runtime.capu, capu_model)?;
    let output = capu.process_text("rồi cũng hỗ trợ cho lâu lâu cũng cho gạo cho này kia")?;
    assert_eq!(
        output,
        "Rồi cũng hỗ trợ cho, lâu lâu cũng cho gạo cho này kia."
    );
    Ok(())
}
