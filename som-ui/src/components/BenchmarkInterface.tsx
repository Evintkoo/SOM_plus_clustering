import { useState } from 'react';
import axios from 'axios';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './BenchmarkInterface.css';

interface DataPoint {
  samples: number;
  time: number;
  throughput: number;
}

export const BenchmarkInterface: React.FC = () => {
  const [chartData, setChartData] = useState<DataPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  const sampleSteps = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000];

  const runBenchmark = async () => {
    setLoading(true);
    setError(null);
    setChartData([]);
    setCurrentStep(0);
    setElapsed(0);

    const startTime = Date.now();
    const timer = setInterval(() => setElapsed(Date.now() - startTime), 100);

    try {
      for (let i = 0; i < sampleSteps.length; i++) {
        const count = sampleSteps[i];
        setCurrentStep(i + 1);
        const response = await axios.post('http://localhost:3000/benchmark', {
          n_samples: count, n_features: 3, grid_size: 10, learning_rate: 0.5,
          neighbor_radius: 3.0, epochs: 10, init_method: 'kmeans_plus_plus', distance: 'euclidean',
          use_gpu: true,
        });
        setChartData(prev => [...prev, {
          samples: count,
          time: response.data.total_time_ms,
          throughput: response.data.samples_per_sec,
        }]);
      }
    } catch {
      setError('Benchmark failed. Ensure the API is running on localhost:3000');
    } finally {
      clearInterval(timer);
      setLoading(false);
    }
  };

  const formatSamples = (v: number) => {
    if (v >= 1_000_000) return `${v / 1_000_000}M`;
    if (v >= 1_000) return `${v / 1_000}K`;
    return String(v);
  };

  const done = !loading && chartData.length > 0;
  const peakThroughput = chartData.length ? Math.max(...chartData.map(d => d.throughput)) : 0;
  const totalTime = chartData.reduce((a, d) => a + d.time, 0);

  return (
    <div className="benchmark">
      <div className="section-eyebrow">
        <span className="section-eyebrow__label">02 / Benchmarks</span>
        <span className="section-eyebrow__line" />
      </div>
      <h2 className="section__title">Performance</h2>
      <p className="section__subtitle">Scaling analysis from 100 to 1M samples — log-log axes for proper complexity visualization</p>

      {/* Config + Run */}
      <div className="card card--config">
        <div className="config-header">
          <h3 className="card__title">Benchmark Configuration</h3>
          <span className="config-badge">
            {sampleSteps.length} steps · {formatSamples(sampleSteps[sampleSteps.length - 1])} max
          </span>
        </div>
        <div className="config-grid">
          <div className="config-item"><span className="config-item__key">Grid</span><span className="config-item__val">10×10</span></div>
          <div className="config-item"><span className="config-item__key">Epochs</span><span className="config-item__val">10</span></div>
          <div className="config-item"><span className="config-item__key">Init</span><span className="config-item__val">KMeans++</span></div>
          <div className="config-item"><span className="config-item__key">Distance</span><span className="config-item__val">Euclidean</span></div>
          <div className="config-item"><span className="config-item__key">Dims</span><span className="config-item__val">3</span></div>
          <div className="config-item"><span className="config-item__key">LR</span><span className="config-item__val">0.5</span></div>
          <div className="config-item"><span className="config-item__key">Backend</span><span className="config-item__val">GPU (Metal)</span></div>
        </div>
        <button className="btn btn--run" onClick={runBenchmark} disabled={loading}>
          {loading ? `Step ${currentStep}/${sampleSteps.length} · ${(elapsed / 1000).toFixed(1)}s` : 'Run Benchmark Suite'}
        </button>
      </div>

      {error && (
        <div className="alert alert--error">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
          <span>{error}</span>
        </div>
      )}

      {/* Live Charts — render once we have data */}
      {chartData.length >= 1 && (
        <>
          {done && (
            <div className="stats-row">
              <div className="stat"><span className="stat__value">{formatSamples(peakThroughput)}</span><span className="stat__label">Peak samples/sec</span></div>
              <div className="stat"><span className="stat__value">{totalTime < 1000 ? `${totalTime.toFixed(0)}ms` : `${(totalTime / 1000).toFixed(1)}s`}</span><span className="stat__label">Total wall time</span></div>
              <div className="stat"><span className="stat__value">{chartData.length}</span><span className="stat__label">Data points</span></div>
              <div className="stat"><span className="stat__value">O(n)</span><span className="stat__label">Expected complexity</span></div>
            </div>
          )}

          <div className="card card--chart">
            <div className="card__gradient-bar" />
            <div className="chart-header">
              <h3 className="card__title">Training Time vs. Sample Size</h3>
              <span className="chart-axis-label">{loading ? `${chartData.length}/${sampleSteps.length} collected` : 'log₁₀(ms) vs log₁₀(n)'}</span>
            </div>
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                  <XAxis dataKey="samples" type="number" stroke="var(--text-tertiary)" fontSize={11} fontFamily="var(--font-mono)" tickFormatter={formatSamples} name="Samples" />
                  <YAxis dataKey="time" type="number" stroke="var(--text-tertiary)" fontSize={11} fontFamily="var(--font-mono)" tickFormatter={(v: number) => v >= 1000 ? `${(v/1000).toFixed(1)}s` : `${v.toFixed(0)}ms`} name="Time" />
                  <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)', borderRadius: 8, fontFamily: 'var(--font-mono)', fontSize: 12 }} formatter={(v: number, name: string) => [name === 'Samples' ? `n = ${Number(v).toLocaleString()}` : (v >= 1000 ? `${(v/1000).toFixed(2)}s` : `${v.toFixed(2)}ms`), name === 'Samples' ? 'Samples' : 'Time']} />
                  <Scatter data={chartData} fill="var(--accent-warm)" line={{ stroke: 'var(--accent-warm)', strokeWidth: 2 }} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card card--chart">
            <div className="card__gradient-bar" />
            <div className="chart-header">
              <h3 className="card__title">Throughput Scaling</h3>
              <span className="chart-axis-label">{loading ? `${chartData.length}/${sampleSteps.length} collected` : 'log₁₀(samples/sec) vs log₁₀(n)'}</span>
            </div>
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                  <XAxis dataKey="samples" type="number" stroke="var(--text-tertiary)" fontSize={11} fontFamily="var(--font-mono)" tickFormatter={formatSamples} name="Samples" />
                  <YAxis dataKey="throughput" type="number" stroke="var(--text-tertiary)" fontSize={11} fontFamily="var(--font-mono)" tickFormatter={formatSamples} name="Throughput" />
                  <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-primary)', borderRadius: 8, fontFamily: 'var(--font-mono)', fontSize: 12 }} formatter={(v: number, name: string) => [name === 'Samples' ? `n = ${Number(v).toLocaleString()}` : `${v.toFixed(2)} samples/sec`, name === 'Samples' ? 'Samples' : 'Throughput']} />
                  <Scatter data={chartData} fill="var(--accent-sage)" line={{ stroke: 'var(--accent-sage)', strokeWidth: 2 }} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          {done && (
            <div className="card card--table">
              <h3 className="card__title">Raw Measurements</h3>
              <div className="table-wrap">
                <table className="data-table">
                  <thead><tr><th>n</th><th>Time (ms)</th><th>Throughput (s⁻¹)</th><th>ms/sample</th></tr></thead>
                  <tbody>
                    {chartData.map((d) => (
                      <tr key={d.samples}>
                        <td>{d.samples.toLocaleString()}</td>
                        <td>{d.time.toFixed(2)}</td>
                        <td>{d.throughput.toFixed(0)}</td>
                        <td>{(d.time / d.samples).toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
