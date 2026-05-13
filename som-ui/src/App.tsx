import { useState, useEffect } from 'react';
import { TrainingInterface } from './components/TrainingInterface';
import { BenchmarkInterface } from './components/BenchmarkInterface';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState<'train' | 'benchmark'>('train');
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    return (document.documentElement.getAttribute('data-theme') as 'light' | 'dark') || 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === 'light' ? 'dark' : 'light');

  return (
    <div className="app">
      <nav className="nav">
        <div className="nav__container">
          <a className="nav__logo" href="#">
            <img className="nav__logo-img nav__logo-img--dark" src="/images/logo-dark.png" alt="SOM++" />
            <img className="nav__logo-img nav__logo-img--light" src="/images/logo-light.png" alt="SOM++" />
            <span className="nav__logo-text">SOM++</span>
          </a>

          <div className="nav__menu">
            <button
              className={`nav__link ${activeTab === 'train' ? 'nav__link--active' : ''}`}
              onClick={() => setActiveTab('train')}
            >
              Training
            </button>
            <button
              className={`nav__link ${activeTab === 'benchmark' ? 'nav__link--active' : ''}`}
              onClick={() => setActiveTab('benchmark')}
            >
              Benchmarks
            </button>
          </div>

          <div className="nav__actions">
            <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
              {theme === 'light' ? (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                </svg>
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="5" />
                  <line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" />
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                  <line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" />
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </nav>

      <main className="main">
        <section className="section">
          {activeTab === 'train' && <TrainingInterface />}
          {activeTab === 'benchmark' && <BenchmarkInterface />}
        </section>
      </main>

      <footer className="footer">
        <div className="footer__top">
          <p className="footer__tagline">Built with curiosity & Rust.</p>
        </div>
        <div className="footer__content">
          <p className="footer__text">© 2025 SOM++ v0.1 — API on <code>localhost:3000</code></p>
          <div className="footer__links">
            <a className="footer__link" href="https://github.com/Evintkoo/SOM_plus_clustering" target="_blank" rel="noopener noreferrer">GitHub</a>
            <a className="footer__link" href="https://docs.rs/som_plus_clustering" target="_blank" rel="noopener noreferrer">Docs</a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
