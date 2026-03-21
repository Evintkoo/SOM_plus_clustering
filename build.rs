fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        compile_cuda_kernels();
    }

    #[cfg(feature = "metal")]
    compile_metal_shaders();
}

fn compile_cuda_kernels() {
    use std::process::Command;
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Write stub PTX files when nvcc is absent so `cargo check --features cuda` still works
    if Command::new("nvcc").arg("--version").output().is_err() {
        println!("cargo:warning=nvcc not found — CUDA feature stubs only");
        for name in &["euclidean_distances", "cosine_distances", "neighborhood_update"] {
            let ptx_path = format!("{}/{}.ptx", out_dir, name);
            std::fs::write(&ptx_path, "// stub\n").unwrap();
        }
        return;
    }

    let shaders_dir = "src/backend/shaders";
    for name in &["euclidean_distances", "cosine_distances", "neighborhood_update"] {
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

#[cfg(feature = "metal")]
fn compile_metal_shaders() {
    use std::process::Command;
    // Gracefully skip if xcrun is absent
    if Command::new("xcrun").args(["--find", "metal"]).status().is_err() {
        println!("cargo:warning=xcrun metal not found — Metal shaders will not be compiled. Requires macOS with Xcode command line tools.");
        return;
    }
    let shaders_dir = "src/backend/shaders";
    for name in &["euclidean_distances", "cosine_distances", "neighborhood_update"] {
        let msl = format!("{}/{}.metal", shaders_dir, name);
        let air = format!("{}/{}.air", shaders_dir, name);
        let lib = format!("{}/{}.metallib", shaders_dir, name);
        Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c", &msl, "-o", &air])
            .status().expect("xcrun metal failed");
        Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib", &air, "-o", &lib])
            .status().expect("xcrun metallib failed");
        println!("cargo:rerun-if-changed={}", msl);
    }
}
