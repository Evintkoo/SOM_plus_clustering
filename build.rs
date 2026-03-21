fn main() {
    #[cfg(feature = "cuda")]
    compile_cuda_kernels();

    #[cfg(feature = "metal")]
    compile_metal_shaders();
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    use std::process::Command;
    // Gracefully skip if nvcc is absent (CI without GPU toolkit)
    if Command::new("nvcc").arg("--version").status().is_err() {
        println!("cargo:warning=nvcc not found — CUDA kernels will not be compiled. Install CUDA toolkit to use the cuda feature.");
        return;
    }
    let shaders_dir = "src/backend/shaders";
    for name in &["euclidean_distances", "cosine_distances", "neighborhood_update"] {
        let cu = format!("{}/{}.cu", shaders_dir, name);
        let ptx = format!("{}/{}.ptx", shaders_dir, name);
        let status = Command::new("nvcc")
            .args(["--ptx", "-o", &ptx, &cu])
            .status()
            .expect("nvcc failed unexpectedly");
        assert!(status.success(), "nvcc failed for {}", cu);
        println!("cargo:rerun-if-changed={}", cu);
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
