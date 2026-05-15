fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        compile_cuda_kernels();
    }

    if std::env::var("CARGO_FEATURE_METAL").is_ok() {
        // Shaders are compiled at runtime via Metal's JIT (new_library_with_source).
        // No build-time xcrun/metal invocation needed.
        println!("cargo:rerun-if-changed=src/backend/shaders/euclidean_distances.metal");
        println!("cargo:rerun-if-changed=src/backend/shaders/neighborhood_update.metal");
    }
}

fn compile_cuda_kernels() {
    use std::process::Command;
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Write stub PTX files when nvcc is absent so `cargo check --features cuda` still works
    if Command::new("nvcc").arg("--version").output().is_err() {
        println!("cargo:warning=nvcc not found — CUDA feature stubs only");
        for name in &[
            "euclidean_distances",
            "cosine_distances",
            "neighborhood_update",
        ] {
            let ptx_path = format!("{}/{}.ptx", out_dir, name);
            std::fs::write(&ptx_path, "// stub\n").unwrap();
        }
        return;
    }

    let shaders_dir = "src/backend/shaders";
    for name in &[
        "euclidean_distances",
        "cosine_distances",
        "neighborhood_update",
    ] {
        let cu_path = format!("{}/{}.cu", shaders_dir, name);
        let ptx_path = format!("{}/{}.ptx", out_dir, name);
        let status = Command::new("nvcc")
            .args(["--ptx", "-o", &ptx_path, &cu_path])
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!("cargo:error=nvcc exited with status {} for {}.cu", s, name);
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("cargo:error=nvcc invocation failed: {}", e);
                std::process::exit(1);
            }
        }
        println!("cargo:rerun-if-changed={}", cu_path);
    }
}
