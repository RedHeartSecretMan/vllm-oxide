use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return Ok(());
    }

    println!("cargo:rerun-if-changed=kernels/reshape_and_cache.cu");
    println!("cargo:rerun-if-changed=kernels/copy_blocks.cu");
    println!("cargo:rerun-if-changed=kernels/sampler.cu");

    let lib = out_dir.join("libvllmoxidekernels.a");

    cudaforge::KernelBuilder::new()
        .source_glob("kernels/*.cu")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--compiler-options")
        .arg("-fPIC")
        .build_lib(lib)
        .map_err(|e| anyhow::anyhow!("cudaforge kernel build failed: {e}"))?;

    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=vllmoxidekernels");
    // These kernels preserve the reference's F32 operation order and must
    // not inherit the fast-math flags used by the cache/sampler kernels.
    println!("cargo:rerun-if-changed=kernels/precision");
    cudaforge::KernelBuilder::new()
        .source_glob("kernels/precision/*.cu")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--compiler-options")
        .arg("-fPIC")
        .build_lib(out_dir.join("libvllmoxideprecision.a"))
        .map_err(|e| anyhow::anyhow!("precision kernel build failed: {e}"))?;
    println!("cargo:rustc-link-lib=vllmoxideprecision");
    println!("cargo:rustc-link-lib=dylib=cudart");

    Ok(())
}
