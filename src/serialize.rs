use crate::SomError;

/// Serializable snapshot of a fitted Som.
#[derive(bincode::Encode, bincode::Decode)]
pub struct SomState {
    pub m: usize,
    pub n: usize,
    pub dim: usize,
    /// Flattened neurons: row-major Vec of length m*n*dim
    pub neurons: Vec<f64>,
    pub initial_lr: f64,
    pub cur_lr: f64,
    pub initial_rad: f64,
    pub cur_rad: f64,
    /// 0 = Euclidean, 1 = Cosine
    pub dist_func: u8,
    pub trained: bool,
}

pub fn save_bincode(state: &SomState, path: &str) -> Result<(), SomError> {
    let bytes = bincode::encode_to_vec(state, bincode::config::standard())
        .map_err(|e| SomError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    std::fs::write(path, bytes)?;
    Ok(())
}

pub fn load_bincode(path: &str) -> Result<SomState, SomError> {
    let bytes = std::fs::read(path)?;
    let (val, _) = bincode::decode_from_slice::<SomState, _>(&bytes, bincode::config::standard())
        .map_err(|e| SomError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    Ok(val)
}

#[cfg(feature = "serde-json")]
pub fn save_json<T: serde::Serialize>(value: &T, path: &str) -> Result<(), SomError> {
    use std::fs::File;
    use std::io::BufWriter;
    let f = File::create(path)?;
    serde_json::to_writer_pretty(BufWriter::new(f), value)
        .map_err(|e| SomError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))
}

#[cfg(feature = "serde-json")]
pub fn load_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T, SomError> {
    use std::fs::File;
    use std::io::BufReader;
    let f = File::open(path)?;
    serde_json::from_reader(BufReader::new(f))
        .map_err(|e| SomError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))
}
