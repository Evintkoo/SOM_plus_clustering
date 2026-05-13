import { useState } from 'react';
import axios from 'axios';
import './TrainingInterface.css';

interface TrainingData {
  data: number[][];
  numSamples: number;
  numDimensions: number;
  gridSize: number;
  learningRate: number;
  neighborRadius: number;
  epochs: number;
  initMethod: string;
  distance: string;
  useGpu: boolean;
}

interface TrainResult {
  model_id: string;
  elapsed_ms: number;
  silhouette: number | null;
  davies_bouldin: number | null;
  calinski_harabasz: number | null;
}

export const TrainingInterface: React.FC = () => {
  const [result, setResult] = useState<TrainResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const [trainingData, setTrainingData] = useState<TrainingData>({
    data: [],
    numSamples: 200,
    numDimensions: 3,
    gridSize: 10,
    learningRate: 0.5,
    neighborRadius: 3.0,
    epochs: 100,
    initMethod: 'kmeans_plus_plus',
    distance: 'euclidean',
    useGpu: false,
  });

  const generateSampleData = () => {
    const data: number[][] = [];
    for (let i = 0; i < trainingData.numSamples; i++) {
      const point: number[] = [];
      for (let d = 0; d < trainingData.numDimensions; d++) {
        point.push(Math.random() * 10);
      }
      data.push(point);
    }
    setTrainingData({ ...trainingData, data });
    setError(null);
  };

  const handleTrain = async () => {
    if (trainingData.data.length === 0) {
      setError('Please generate or upload data first');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setElapsed(0);

    const startTime = Date.now();
    const timer = setInterval(() => setElapsed(Date.now() - startTime), 100);

    try {
      setProgress('Sending data to server...');
      await new Promise(r => setTimeout(r, 100));

      setProgress(`Training ${trainingData.data.length.toLocaleString()} samples (${trainingData.epochs} epochs)...`);
      const response = await axios.post('http://localhost:3000/train', {
        data: trainingData.data,
        grid_size: trainingData.gridSize,
        learning_rate: trainingData.learningRate,
        neighbor_radius: trainingData.neighborRadius,
        epochs: trainingData.epochs,
        init_method: trainingData.initMethod,
        distance: trainingData.distance,
        use_gpu: trainingData.useGpu,
      });

      setProgress('Computing evaluation metrics...');
      await new Promise(r => setTimeout(r, 200));

      setResult({
        model_id: response.data.model_id,
        elapsed_ms: response.data.elapsed_ms,
        silhouette: response.data.silhouette,
        davies_bouldin: response.data.davies_bouldin,
        calinski_harabasz: response.data.calinski_harabasz,
      });
      setProgress('');
    } catch {
      setError('Training failed. Make sure the API is running on port 3000.');
      setProgress('');
    } finally {
      clearInterval(timer);
      setLoading(false);
    }
  };

  return (
    <div className="training">
      <div className="section-eyebrow">
        <span className="section-eyebrow__label">01 / Training</span>
        <span className="section-eyebrow__line" />
      </div>
      <h2 className="section__title">Configure & Train</h2>
      <p className="section__subtitle">Set hyperparameters and train a Self-Organizing Map</p>

      <div className="card card--controls">
        <div className="card__header">
          <button className="btn btn--primary" onClick={generateSampleData}>
            Generate Sample Data
            <span className="btn__badge">{trainingData.numSamples} × {trainingData.numDimensions}d</span>
          </button>
        </div>

        <div className="param-grid">
          <div className="param">
            <label className="param__label">Samples</label>
            <input
              className="param__input"
              type="number"
              value={trainingData.numSamples}
              onChange={(e) => setTrainingData({ ...trainingData, numSamples: parseInt(e.target.value) })}
              min="10" max="10000"
            />
          </div>
          <div className="param">
            <label className="param__label">Dimensions</label>
            <input
              className="param__input"
              type="number"
              value={trainingData.numDimensions}
              onChange={(e) => setTrainingData({ ...trainingData, numDimensions: parseInt(e.target.value) })}
              min="2" max="100"
            />
          </div>
          <div className="param">
            <label className="param__label">Grid Size</label>
            <input
              className="param__input"
              type="number"
              value={trainingData.gridSize}
              onChange={(e) => setTrainingData({ ...trainingData, gridSize: parseInt(e.target.value) })}
              min="5" max="50"
            />
          </div>
          <div className="param">
            <label className="param__label">Learning Rate</label>
            <input
              className="param__input"
              type="number"
              value={trainingData.learningRate}
              onChange={(e) => setTrainingData({ ...trainingData, learningRate: parseFloat(e.target.value) })}
              step="0.1" min="0.1" max="1.0"
            />
          </div>
          <div className="param">
            <label className="param__label">Neighbor Radius</label>
            <input
              className="param__input"
              type="number"
              value={trainingData.neighborRadius}
              onChange={(e) => setTrainingData({ ...trainingData, neighborRadius: parseFloat(e.target.value) })}
              step="0.5" min="0.5"
            />
          </div>
          <div className="param">
            <label className="param__label">Epochs</label>
            <input
              className="param__input"
              type="number"
              value={trainingData.epochs}
              onChange={(e) => setTrainingData({ ...trainingData, epochs: parseInt(e.target.value) })}
              min="10" max="500"
            />
          </div>
          <div className="param">
            <label className="param__label">Init Method</label>
            <select
              className="param__input"
              value={trainingData.initMethod}
              onChange={(e) => setTrainingData({ ...trainingData, initMethod: e.target.value })}
            >
              <option value="random">Random</option>
              <option value="kmeans">KMeans</option>
              <option value="kmeans_plus_plus">KMeans++</option>
              <option value="kde">KDE</option>
              <option value="som_plus_plus">SOM++</option>
            </select>
          </div>
          <div className="param">
            <label className="param__label">Distance</label>
            <select
              className="param__input"
              value={trainingData.distance}
              onChange={(e) => setTrainingData({ ...trainingData, distance: e.target.value })}
            >
              <option value="euclidean">Euclidean</option>
              <option value="cosine">Cosine</option>
              <option value="manhattan">Manhattan</option>
            </select>
          </div>
          <div className="param">
            <label className="param__label">Backend</label>
            <select
              className="param__input"
              value={trainingData.useGpu ? 'gpu' : 'cpu'}
              onChange={(e) => setTrainingData({ ...trainingData, useGpu: e.target.value === 'gpu' })}
            >
              <option value="cpu">CPU (rayon)</option>
              <option value="gpu">GPU (Metal)</option>
            </select>
          </div>
        </div>

        <button className="btn btn--train" onClick={handleTrain} disabled={loading}>
          {loading ? `Training... (${(elapsed / 1000).toFixed(1)}s)` : 'Train Model'}
        </button>

        {loading && progress && (
          <div className="progress-bar">
            <div className="progress-bar__track">
              <div className="progress-bar__fill" />
            </div>
            <span className="progress-bar__text">{progress}</span>
          </div>
        )}
      </div>

      {error && (
        <div className="alert alert--error">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div className="card card--success">
          <div className="card__gradient-bar" />
          <h3 className="card__title"><svg className="icon icon--success" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 6L9 17l-5-5"/></svg> Model Trained</h3>
          <div className="result-grid">
            <div className="result-item">
              <span className="result-item__label">Model ID</span>
              <code className="result-item__value">{result.model_id}</code>
            </div>
            <div className="result-item">
              <span className="result-item__label">Training Time</span>
              <span className="result-item__value">{result.elapsed_ms < 1000 ? `${result.elapsed_ms.toFixed(1)} ms` : `${(result.elapsed_ms / 1000).toFixed(2)} s`}</span>
            </div>
            <div className="result-item">
              <span className="result-item__label">Throughput</span>
              <span className="result-item__value">{(trainingData.data.length / (result.elapsed_ms / 1000)).toFixed(0)} samples/s</span>
            </div>
            <div className="result-item">
              <span className="result-item__label">Samples</span>
              <span className="result-item__value">{trainingData.data.length.toLocaleString()}</span>
            </div>
            <div className="result-item">
              <span className="result-item__label">Dimensions</span>
              <span className="result-item__value">{trainingData.numDimensions}</span>
            </div>
            <div className="result-item">
              <span className="result-item__label">Grid</span>
              <span className="result-item__value">{trainingData.gridSize}×{trainingData.gridSize}</span>
            </div>
            <div className="result-item">
              <span className="result-item__label">Neurons</span>
              <span className="result-item__value">{trainingData.gridSize * trainingData.gridSize}</span>
            </div>
            <div className="result-item">
              <span className="result-item__label">Total Iterations</span>
              <span className="result-item__value">{(trainingData.epochs * trainingData.data.length).toLocaleString()}</span>
            </div>
            <div className="result-item">
              <span className="result-item__label">ms/sample</span>
              <span className="result-item__value">{(result.elapsed_ms / trainingData.data.length).toFixed(4)}</span>
            </div>
          </div>

          <h4 className="result-section-title">Evaluation Metrics</h4>
          <div className="result-grid">
            <div className="result-item">
              <span className="result-item__label">Silhouette</span>
              <span className="result-item__value result-item__value--metric">{result.silhouette != null ? result.silhouette.toFixed(4) : '—'}</span>
            </div>
            <div className="result-item">
              <span className="result-item__label">Davies-Bouldin</span>
              <span className="result-item__value result-item__value--metric">{result.davies_bouldin != null ? result.davies_bouldin.toFixed(4) : '—'}</span>
            </div>
            <div className="result-item">
              <span className="result-item__label">Calinski-Harabasz</span>
              <span className="result-item__value result-item__value--metric">{result.calinski_harabasz != null ? result.calinski_harabasz.toFixed(2) : '—'}</span>
            </div>
          </div>
        </div>
      )}

      <div className="card card--info">
        <h3 className="card__title">Optimizations Enabled</h3>
        <div className="opt-grid">
          {[
            { svg: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>, label: 'Fast Inverse Sqrt', speed: '2-3×' },
            { svg: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="13 17 18 12 13 7"/><polyline points="6 17 11 12 6 7"/></svg>, label: 'Taylor Exp()', speed: '2-4×' },
            { svg: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>, label: 'Gaussian Caching', speed: '50-70%' },
            { svg: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>, label: 'Norm Caching', speed: '25-35%' },
            { svg: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>, label: 'Squared Distance', speed: '1.5×' },
            { svg: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="3 11 22 2 13 21 11 13 3 11"/></svg>, label: 'Manhattan Dist', speed: 'New' },
          ].map((opt) => (
            <div className="opt-item" key={opt.label}>
              <span className="opt-item__icon">{opt.svg}</span>
              <span className="opt-item__label">{opt.label}</span>
              <span className="opt-item__speed">{opt.speed}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
