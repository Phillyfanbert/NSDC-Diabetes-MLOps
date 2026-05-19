"""
generate_dashboard.py — generates dashboard.html in the project root
=====================================================================
The dashboard fetches live model metrics from the /health endpoint
on page load, so it always reflects the current Production model
without needing to be regenerated after each retrain.

Usage:
    python src/generate_dashboard.py
    open dashboard.html          # macOS (run while serve_model.py is running)
"""

from pathlib import Path

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Diabetes MLOps Dashboard</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
      background: #f5f5f3;
      color: #1a1a1a;
      padding: 2rem;
      max-width: 880px;
      margin: 0 auto;
      line-height: 1.5;
    }

    @media (prefers-color-scheme: dark) {
      body { background: #1a1a1a; color: #e8e8e6; }
      .card { background: #242422; border-color: rgba(255,255,255,0.1); }
      .metric-card { background: #2c2c2a; }
      .coef-bar-wrap, .gauge-bar { background: rgba(255,255,255,0.08); }
      code { background: #2c2c2a; color: #a8a89e; }
      .info-box { background: #1e2a1e; border-color: rgba(100,180,100,0.2); }
      .info-box p { color: #88c888; }
      .warn-box { background: #2a1e1e; border-color: rgba(180,80,80,0.2); }
      .warn-box p { color: #e88888; }
      .divider-row { border-color: rgba(255,255,255,0.08); }
      input[type=range] { accent-color: #378ADD; }
      .predict-btn { background: #2c2c2a; color: #e8e8e6; border-color: rgba(255,255,255,0.15); }
      .predict-btn:hover { background: #363634; }
      .result-box { background: #2c2c2a; border-color: rgba(255,255,255,0.1); }
      .error-box { background: #2a1e1e; border-color: rgba(180,80,80,0.3); }
      .error-box p { color: #e88888; }
    }

    .section-label {
      font-size: 11px; font-weight: 500; color: #888780;
      letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 10px;
    }
    .card {
      background: #fff; border: 0.5px solid rgba(0,0,0,0.12);
      border-radius: 12px; padding: 1rem 1.25rem;
    }
    .metric-card { background: #f1efe8; border-radius: 8px; padding: 1rem; }
    .metric-label { font-size: 12px; color: #888780; margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: 500; }
    .metric-sub   { font-size: 12px; color: #b4b2a9; margin-top: 2px; }
    .metric-loading { font-size: 18px; color: #b4b2a9; }

    .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 600px) {
      .grid-4 { grid-template-columns: repeat(2, 1fr); }
      .grid-2 { grid-template-columns: 1fr; }
    }

    .slider-row { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
    .slider-row label { font-size: 13px; color: #5f5e5a; width: 145px; flex-shrink: 0; }
    .slider-row input[type=range] { flex: 1; accent-color: #378ADD; }
    .slider-row .val { font-size: 13px; font-weight: 500; width: 44px; text-align: right; }

    .predict-btn {
      width: 100%; padding: 10px; font-size: 14px; font-weight: 500;
      cursor: pointer; border-radius: 8px; border: 0.5px solid rgba(0,0,0,0.2);
      background: #fff; color: #1a1a1a; margin-top: 4px; transition: background 0.15s;
    }
    .predict-btn:hover { background: #f1efe8; }

    .result-box {
      margin-top: 14px; border-radius: 8px; padding: 14px 16px;
      border: 0.5px solid rgba(0,0,0,0.1); background: #f1efe8; display: none;
    }
    .result-number { font-size: 36px; font-weight: 500; }
    .result-label  { font-size: 13px; color: #888780; margin-top: 2px; }

    .gauge-bar  { height: 8px; border-radius: 4px; background: #d3d1c7; margin-top: 12px; overflow: hidden; }
    .gauge-fill { height: 100%; border-radius: 4px; background: #378ADD; transition: width 0.5s ease; }
    .range-labels { display: flex; justify-content: space-between; font-size: 11px; color: #b4b2a9; margin-top: 4px; }

    .badge { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 6px; font-weight: 500; }
    .badge-success { background: #eaf3de; color: #3b6d11; }
    .badge-info    { background: #e6f1fb; color: #185fa5; }
    .badge-warning { background: #faeeda; color: #854f0b; }
    .badge-danger  { background: #fcebeb; color: #a32d2d; }
    .badge-offline { background: #d3d1c7; color: #5f5e5a; }

    .divider-row {
      display: flex; justify-content: space-between; align-items: center;
      padding: 8px 0; border-bottom: 0.5px solid rgba(0,0,0,0.08); font-size: 13px;
    }
    .divider-row:last-child { border-bottom: none; }
    .divider-row span:first-child { color: #888780; }
    .divider-row span:last-child  { font-weight: 500; }

    .info-box {
      margin-top: 12px; padding: 10px 14px; border-radius: 8px;
      background: #eaf3de; border: 0.5px solid rgba(60,130,60,0.2);
    }
    .info-box p { font-size: 12px; color: #3b6d11; }

    .warn-box {
      margin-top: 12px; padding: 10px 14px; border-radius: 8px;
      background: #fcebeb; border: 0.5px solid rgba(180,60,60,0.2);
    }
    .warn-box p { font-size: 12px; color: #a32d2d; }

    .error-box { display: none; margin-top: 12px; padding: 10px 14px; border-radius: 8px; border: 0.5px solid rgba(180,60,60,0.3); background: #fcebeb; }
    .error-box p { font-size: 13px; color: #a32d2d; }

    code {
      font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px;
      background: #f1efe8; padding: 1px 5px; border-radius: 4px;
    }
    .header-row {
      display: flex; align-items: baseline; gap: 12px;
      margin-bottom: 1.5rem; flex-wrap: wrap;
    }
    .header-title { font-size: 20px; font-weight: 500; }
    .header-meta  { font-size: 12px; color: #b4b2a9; margin-left: auto; }

    .split-note {
      font-size: 11px; color: #b4b2a9; margin-top: 6px; font-style: italic;
    }
  </style>
</head>
<body>

  <div class="header-row">
    <span class="header-title">Diabetes prevalence predictor</span>
    <span class="badge" id="model-status-badge">Connecting…</span>
    <span class="header-meta">NSDC MLOps &middot; WHO GHO data</span>
  </div>

  <!-- ── Live model performance (populated from /health) ───────────── -->
  <p class="section-label">Model performance <span style="font-size:10px; font-weight:400; text-transform:none; letter-spacing:0;">(live from API)</span></p>
  <div class="grid-4" style="margin-bottom: 1.5rem;">
    <div class="metric-card">
      <div class="metric-label">Test R&sup2; <span style="font-size:10px;">(held-out)</span></div>
      <div class="metric-value" id="m-test-r2"><span class="metric-loading">…</span></div>
      <div class="metric-sub" id="m-test-r2-sub">Loading…</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Train R&sup2; <span style="font-size:10px;">(in-sample)</span></div>
      <div class="metric-value" id="m-train-r2"><span class="metric-loading">…</span></div>
      <div class="metric-sub" id="m-train-r2-sub">Loading…</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Test RMSE</div>
      <div class="metric-value" id="m-rmse"><span class="metric-loading">…</span></div>
      <div class="metric-sub">Avg prediction error (pp)</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Model type</div>
      <div class="metric-value" id="m-version" style="font-size: 15px; padding-top: 4px;"><span class="metric-loading">…</span></div>
      <div class="metric-sub" id="m-version-sub">Loading…</div>
    </div>
  </div>

  <!-- ── Overfit warning ───────────────────────────────────────────── -->
  <div class="warn-box" style="margin-bottom: 1.5rem; display:none;" id="overfit-warning">
    <p>⚠️  Train R² is notably higher than Test R² — the model may be overfitting on the training years.
    This is expected for linear regression with highly collinear lag features.</p>
  </div>

  <!-- ── Promotion reason (populated from /health) ─────────────────── -->
  <div class="info-box" style="margin-bottom: 1.5rem; display:none;" id="promotion-box">
    <p id="promotion-reason"></p>
  </div>

  <div class="grid-2" style="margin-bottom: 1.5rem;">

    <div class="card">
      <p class="section-label">Global benchmarks</p>
      <p style="font-size: 12px; color: #888780; margin-bottom: 14px;">Reference points from WHO dataset</p>
      <div class="divider-row"><span>Global avg diabetes</span><span>7.9%</span></div>
      <div class="divider-row"><span>Global avg obesity</span><span>14.9%</span></div>
      <div class="divider-row"><span>Highest diabetes (Pacific)</span><span style="color: #a32d2d;">~30%</span></div>
      <div class="divider-row"><span>Lag window</span><span>1, 2, 3 years</span></div>
      <div class="divider-row"><span>CV R&sup2; (5-fold)</span><span id="m-cv-r2">…</span></div>
      <div class="divider-row"><span>Train / test split year</span><span id="m-split-year">…</span></div>
      <p class="split-note">Chronological split — test set = years &ge; <span id="m-split-year-note">…</span> (never seen during training)</p>
    </div>

    <div class="card">
      <p class="section-label">Make a prediction</p>
      <p style="font-size: 13px; color: #5f5e5a; margin-bottom: 1rem;">
        Enter the current obesity rate and how fast it is rising to predict diabetes prevalence.
        Calls <code>POST /predict</code> on the local API.
      </p>

      <div class="slider-row">
        <label>Current obesity %</label>
        <input type="range" min="1" max="60" step="0.5" value="28.5" id="s0" oninput="sync(0,this.value)">
        <span class="val" id="v0">28.5</span>
      </div>
      <div class="slider-row">
        <label>Obesity trend (pp/yr)</label>
        <input type="range" min="-2" max="2" step="0.05" value="0.45" id="s1" oninput="sync(1,this.value)">
        <span class="val" id="v1">0.45</span>
      </div>

      <button class="predict-btn" onclick="runPredict()">
        <i class="ti ti-brain" aria-hidden="true"></i>
        Predict diabetes prevalence
      </button>

      <div class="result-box" id="result-box">
        <div style="display:flex; align-items:baseline; gap:10px;">
          <span class="result-number" id="result-number">—</span>
          <span style="font-size:18px; color:#888780;">%</span>
          <span class="badge" id="result-badge" style="margin-left:4px;"></span>
        </div>
        <div class="result-label">Predicted diabetes prevalence</div>
        <div class="gauge-bar">
          <div class="gauge-fill" id="gauge-fill" style="width:0%;"></div>
        </div>
        <div class="range-labels"><span>0%</span><span>Global avg 7.9%</span><span>30%</span></div>
        <p style="font-size:12px; color:#888780; margin-top:10px;" id="result-note"></p>
      </div>

      <div class="error-box" id="error-box">
        <p id="error-msg"></p>
      </div>
    </div>

  </div>

  <div class="info-box" id="api-info-box">
    <p>
      <i class="ti ti-info-circle" aria-hidden="true" style="font-size:14px; vertical-align:-2px; margin-right:4px;"></i>
      Dashboard calls <code>http://127.0.0.1:8000</code>.
      Make sure <code>python src/serve_model.py</code> is running.
    </p>
  </div>

  <script>
    const API = 'http://127.0.0.1:8000';
    const vals = [28.5, 27.1, 25.8, 24.3];

    // ── Load live metrics from /health on page load ─────────────────────────
    async function loadHealth() {
      try {
        const res  = await fetch(API + '/health');
        const data = await res.json();

        const testR2    = data.test_r2    ?? null;
        const trainR2   = data.train_r2   ?? null;
        const testRmse  = data.test_rmse  ?? null;
        const cvR2      = data.cv_r2      ?? null;
        const version   = data.version    ?? '?';
        const splitYear = data.split_year ?? null;

        // Status badge
        const badge = document.getElementById('model-status-badge');
        if (data.model_loaded && data.scale_params_loaded) {
          badge.textContent = 'Model live';
          badge.className   = 'badge badge-success';
        } else {
          badge.textContent = 'Model offline';
          badge.className   = 'badge badge-danger';
        }

        // Test R²
        if (testR2 !== null) {
          document.getElementById('m-test-r2').textContent = testR2.toFixed(4);
          const pct = (testR2 * 100).toFixed(1);
          document.getElementById('m-test-r2-sub').textContent =
            `Explains ${pct}% of variance (held-out)`;
        }

        // Train R²
        if (trainR2 !== null) {
          document.getElementById('m-train-r2').textContent = trainR2.toFixed(4);
          const pct = (trainR2 * 100).toFixed(1);
          document.getElementById('m-train-r2-sub').textContent =
            `Explains ${pct}% of variance (in-sample)`;
        }

        // Show overfit warning if train R² is >0.05 higher than test R²
        if (trainR2 !== null && testR2 !== null && (trainR2 - testR2) > 0.05) {
          document.getElementById('overfit-warning').style.display = 'block';
        }

        // RMSE
        if (testRmse !== null) {
          document.getElementById('m-rmse').textContent = testRmse.toFixed(4) + '%';
        }

        // Model type + version
        const modelType = data.model_type ?? 'Unknown';
        const modelLabel = modelType === 'Ridge'
          ? 'Ridge Regression'
          : modelType === 'LinearRegression'
          ? 'Linear Regression'
          : modelType;
        document.getElementById('m-version').textContent = modelLabel;
        document.getElementById('m-version-sub').textContent = modelType === 'Ridge'
          ? 'v' + version + ' · L2 regularised · 1/2/3yr lags'
          : 'v' + version + ' · 1/2/3yr lags';

        // Promotion reason
        const reason = data.promotion_reason ?? null;
        if (reason) {
          document.getElementById('promotion-reason').textContent = '📋 ' + reason;
          document.getElementById('promotion-box').style.display = 'block';
        }

        // CV R²
        if (cvR2 !== null) {
          document.getElementById('m-cv-r2').textContent = cvR2.toFixed(4);
        }

        // Split year — populated live from /health so it stays accurate
        // if the data shifts the 80/20 boundary on retrain.
        // Falls back to '?' if the API doesn't expose split_year yet
        // (requires serve_model.py to include it in the /health response).
        if (splitYear !== null) {
          document.getElementById('m-split-year').textContent      = splitYear;
          document.getElementById('m-split-year-note').textContent = splitYear;
        } else {
          document.getElementById('m-split-year').textContent      = '?';
          document.getElementById('m-split-year-note').textContent = '?';
        }

      } catch (e) {
        // API not reachable
        const badge = document.getElementById('model-status-badge');
        badge.textContent = 'API offline';
        badge.className   = 'badge badge-offline';
        ['m-test-r2', 'm-train-r2', 'm-rmse', 'm-version'].forEach(id => {
          document.getElementById(id).textContent = '—';
        });
        document.getElementById('m-test-r2-sub').textContent  = 'Start serve_model.py';
        document.getElementById('m-train-r2-sub').textContent = 'Start serve_model.py';
        document.getElementById('m-cv-r2').textContent        = '—';
        document.getElementById('m-split-year').textContent      = '—';
        document.getElementById('m-split-year-note').textContent = '?';
      }
    }

    // ── Slider sync ─────────────────────────────────────────────────────────
    function sync(i, v) {
      vals[i] = parseFloat(v);
      document.getElementById('v' + i).textContent = parseFloat(v).toFixed(1);
    }

    function classify(v) {
      if (v < 5)  return ['Low',            'badge-success'];
      if (v < 9)  return ['Near global avg','badge-info'];
      if (v < 15) return ['Elevated',       'badge-warning'];
      return             ['High',           'badge-danger'];
    }

    // ── Prediction ──────────────────────────────────────────────────────────
    async function runPredict() {
      document.getElementById('error-box').style.display  = 'none';
      document.getElementById('result-box').style.display = 'none';

      const body = {
        obesity_current: vals[0],
        obesity_trend:   vals[1]
      };

      try {
        const res = await fetch(API + '/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'API error ' + res.status);
        }

        const data = await res.json();
        const pct  = parseFloat(data.predicted_diabetes_prevalence_pct);
        const [label, badgeClass] = classify(pct);

        document.getElementById('result-number').textContent = pct.toFixed(2);

        const badge = document.getElementById('result-badge');
        badge.textContent = label;
        badge.className   = 'badge ' + badgeClass;

        const gaugeWidth = Math.min((pct / 30) * 100, 100);
        document.getElementById('gauge-fill').style.width = gaugeWidth.toFixed(1) + '%';

        const diff = (pct - 7.9).toFixed(2);
        const dir  = diff >= 0 ? 'above' : 'below';
        document.getElementById('result-note').textContent =
          Math.abs(diff) + '% ' + dir + ' the global average of 7.9%.';

        document.getElementById('result-box').style.display = 'block';

      } catch (e) {
        document.getElementById('error-msg').textContent =
          'Could not reach the API: ' + e.message + '. Is serve_model.py running?';
        document.getElementById('error-box').style.display = 'block';
      }
    }

    // Load health metrics immediately on page open
    loadHealth();
  </script>

</body>
</html>"""


def main():
    out = Path("dashboard.html")
    out.write_text(HTML, encoding="utf-8")
    print(f"✅ Dashboard written to {out.resolve()}")
    print("   Open it in your browser while serve_model.py is running.")


if __name__ == "__main__":
    main()