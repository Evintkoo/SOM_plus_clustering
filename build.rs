fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        compile_cuda_kernels();
    }

    if std::env::var("CARGO_FEATURE_METAL").is_ok() {
        compile_metal_shaders();
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

fn compile_metal_shaders() {
    use std::process::Command;
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let metallib_out = format!("{}/som_kernels.metallib", out_dir);

    // Write a stub metallib when xcrun/metal is absent (e.g. Linux CI) so that
    // `include_bytes!` in metal.rs compiles. The stub will cause a runtime error
    // when the Metal device is initialised, which is the correct behaviour on
    // non-Apple hardware.
    let xcrun_available = Command::new("xcrun")
        .args(["--find", "metal"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !xcrun_available {
        println!("cargo:warning=xcrun metal not found — writing stub metallib. Requires macOS with Xcode command line tools.");
        // Minimal valid-looking stub so include_bytes! succeeds; Metal device init
        // will fail at runtime anyway on non-Apple hardware.
        std::fs::write(&metallib_out, b"MTLB").unwrap();
        return;
    }

    let shaders_dir = "src/backend/shaders";
    let msl_files = ["euclidean_distances", "neighborhood_update"];

    // Compile each .metal to .air
    let mut air_files: Vec<String> = Vec::new();
    for name in &msl_files {
        let msl = format!("{}/{}.metal", shaders_dir, name);
        let air = format!("{}/{}.air", out_dir, name);
        let status = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c", &msl, "-o", &air])
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!(
                    "cargo:error=xcrun metal -c failed for {}.metal with status {}",
                    name, s
                );
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("cargo:error=xcrun invocation failed: {}", e);
                std::process::exit(1);
            }
        }
        air_files.push(air);
        println!("cargo:rerun-if-changed={}", msl);
    }

    // Link all .air files into a single som_kernels.metallib
    let mut metallib_args = vec![
        "-sdk".to_string(),
        "macosx".to_string(),
        "metallib".to_string(),
    ];
    metallib_args.extend(air_files);
    metallib_args.extend(["-o".to_string(), metallib_out]);
    let status = Command::new("xcrun").args(&metallib_args).status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            eprintln!("cargo:error=xcrun metallib failed with status {}", s);
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("cargo:error=xcrun metallib invocation failed: {}", e);
            std::process::exit(1);
        }
    }
}
