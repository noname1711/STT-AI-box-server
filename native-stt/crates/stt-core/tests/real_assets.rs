use std::path::PathBuf;

use anyhow::Result;

use stt_core::SttRuntime;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

#[test]
fn probe_matches_locked_baseline() -> Result<()> {
    let runtime = SttRuntime::load(Some(workspace_root()), None)?;
    let probe = runtime.run_probe(Some("vit_stt_vi_v2"))?;
    assert!(probe.matches_expected);
    assert_eq!(probe.actual, "ĐỊNH NGHĨA THẾ NÀO LÀ ĂN MẶC ĐẸP");
    Ok(())
}
