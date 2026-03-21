fn main() {
    #[cfg(feature = "cuda")]
    compile_cuda_kernels();

    #[cfg(feature = "metal")]
    compile_metal_shaders();
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    use std::process::Command;
    let shaders_dir = "src/backend/shaders";
    for name in &["euclidean_distances", "cosine_distances", "neighborhood_update"] {
        let cu = format!("{}/{}.cu", shaders_dir, name);
        let ptx = format!("{}/{}.ptx", shaders_dir, name);
        let status = Command::new("nvcc")
            .args(["--ptx", "-o", &ptx, &cu])
            .status()
            .expect("nvcc not found — install CUDA toolkit");
        assert!(status.success(), "nvcc failed for {}", cu);
        println!("cargo:rerun-if-changed={}", cu);
    }
}

#[cfg(feature = "metal")]
fn compile_metal_shaders() {
    use std::process::Command;
    let shaders_dir = "src/backend/shaders";
    for name in &["euclidean_distances", "cosine_distances", "neighborhood_update"] {
        let msl = format!("{}/{}.metal", shaders_dir, name);
        let air = format!("{}/{}.air", shaders_dir, name);
        let lib = format!("{}/{}.metallib", shaders_dir, name);
        Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c", &msl, "-o", &air])
            .status().expect("xcrun metal not found");
        Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib", &air, "-o", &lib])
            .status().expect("xcrun metallib failed");
        println!("cargo:rerun-if-changed={}", msl);
    }
}
