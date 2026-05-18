"""
generate_dashboard.py — generates dashboard.html in the project root
=====================================================================
Run once to create the dashboard. Any team member can regenerate it.

Usage:
    python src/generate_dashboard.py
    open dashboard.html          # macOS
    start dashboard.html         # Windows
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
      .divider-row { border-color: rgba(255,255,255,0.08); }
      input[type=range] { accent-color: #378ADD; }
      .predict-btn { background: #2c2c2a; color: #e8e8e6; border-color: rgba(255,255,255,0.15); }
      .predict-btn:hover { background: #363634; }
      .result-box { background: #2c2c2a; border-color: rgba(255,255,255,0.1); }
      .error-box { background: #2a1e1e; border-color: rgba(180,80,80,0.3); }
      .error-box p { color: #e88888; }
    }

    .section-label {
      font-size: 11px;
      font-weight: 500;
      color: #888780;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }

    .card {
      background: #fff;
      border: 0.5px solid rgba(0,0,0,0.12);
      border-radius: 12px;
      padding: 1rem 1.25rem;
    }

    .metric-card {
      background: #f1efe8;
      border-radius: 8px;
      padding: 1rem;
    }

    .metric-label { font-size: 12px; color: #888780; margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: 500; }
    .metric-sub { font-size: 12px; color: #b4b2a9; margin-top: 2px; }

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
      width: 100%;
      padding: 10px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      border-radius: 8px;
      border: 0.5px solid rgba(0,0,0,0.2);
      background: #fff;
      color: #1a1a1a;
      margin-top: 4px;
      transition: background 0.15s;
    }
    .predict-btn:hover { background: #f1efe8; }
    .predict-btn:active { transform: scale(0.99); }

    .result-box {
      margin-top: 14px;
      border-radius: 8px;
      padding: 14px 16px;
      border: 0.5px solid rgba(0,0,0,0.1);
      background: #f1efe8;
      display: none;
    }

    .result-number { font-size: 36px; font-weight: 500; }
    .result-label { font-size: 13px; color: #888780; margin-top: 2px; }

    .gauge-bar { height: 8px; border-radius: 4px; background: #d3d1c7; margin-top: 12px; overflow: hidden; }
    .gauge-fill { height: 100%; border-radius: 4px; background: #378ADD; transition: width 0.5s ease; }
    .range-labels { display: flex; justify-content: space-between; font-size: 11px; color: #b4b2a9; margin-top: 4px; }

    .badge {
      display: inline-block;
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 6px;
      font-weight: 500;
    }
    .badge-success { background: #eaf3de; color: #3b6d11; }
    .badge-info    { background: #e6f1fb; color: #185fa5; }
    .badge-warning { background: #faeeda; color: #854f0b; }
    .badge-danger  { background: #fcebeb; color: #a32d2d; }

    .coef-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .coef-name { font-size: 12px; color: #5f5e5a; width: 160px; flex-shrink: 0; }
    .coef-bar-wrap { flex: 1; height: 8px; background: #d3d1c7; border-radius: 4px; overflow: hidden; }
    .coef-bar { height: 100%; border-radius: 4px; }
    .coef-val { font-size: 12px; font-weight: 500; width: 36px; text-align: right; }

    .divider-row {
      display: flex; justify-content: space-between; align-items: center;
      padding: 8px 0;
      border-bottom: 0.5px solid rgba(0,0,0,0.08);
      font-size: 13px;
    }
    .divider-row:last-child { border-bottom: none; }
    .divider-row span:first-child { color: #888780; }
    .divider-row span:last-child { font-weight: 500; }

    .info-box {
      margin-top: 12px;
      padding: 10px 14px;
      border-radius: 8px;
      background: #eaf3de;
      border: 0.5px solid rgba(60,130,60,0.2);
    }
    .info-box p { font-size: 12px; color: #3b6d11; }

    .error-box {
      display: none;
      margin-top: 12px;
      padding: 10px 14px;
      border-radius: 8px;
      border: 0.5px solid rgba(180,60,60,0.3);
      background: #fcebeb;
    }
    .error-box p { font-size: 13px; color: #a32d2d; }

    code {
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 12px;
      background: #f1efe8;
      padding: 1px 5px;
      border-radius: 4px;
    }

    .header-row {
      display: flex;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 1.5rem;
      flex-wrap: wrap;
    }
    .header-title { font-size: 20px; font-weight: 500; }
    .header-meta { font-size: 12px; color: #b4b2a9; margin-left: auto; }
  </style>
</head>
<body>

  <div class="header-row">
    <span class="header-title">Diabetes prevalence predictor</span>
    <span class="badge badge-success">Model live</span>
    <span class="header-meta">NSDC MLOps &middot; WHO GHO data</span>
  </div>

  <p class="section-label">Model performance</p>
  <div class="grid-4" style="margin-bottom: 1.5rem;">
    <div class="metric-card">
      <div class="metric-label">R&sup2; score</div>
      <div class="metric-value">0.699</div>
      <div class="metric-sub">Explains 70% of variance</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">RMSE</div>
      <div class="metric-value">2.17%</div>
      <div class="metric-sub">Avg prediction error</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Training data</div>
      <div class="metric-value">5,075</div>
      <div class="metric-sub">Country-year observations</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Model type</div>
      <div class="metric-value" style="font-size: 15px; padding-top: 4px;">Linear regression</div>
      <div class="metric-sub">Baseline &middot; 1/2/3yr lags</div>
    </div>
  </div>

  <div class="grid-2" style="margin-bottom: 1.5rem;">

    <div class="card">
      <p class="section-label">Feature coefficients</p>
      <p style="font-size: 12px; color: #888780; margin-bottom: 14px;">How much each obesity lag drives the prediction</p>
      <div class="coef-row">
        <span class="coef-name">Current obesity</span>
        <div class="coef-bar-wrap"><div class="coef-bar" style="width:72%; background:#378ADD;"></div></div>
        <span class="coef-val">+2.84</span>
      </div>
      <div class="coef-row">
        <span class="coef-name">Obesity 1 yr ago</span>
        <div class="coef-bar-wrap"><div class="coef-bar" style="width:58%; background:#5DCAA5;"></div></div>
        <span class="coef-val">+2.29</span>
      </div>
      <div class="coef-row">
        <span class="coef-name">Obesity 2 yrs ago</span>
        <div class="coef-bar-wrap"><div class="coef-bar" style="width:38%; background:#5DCAA5;"></div></div>
        <span class="coef-val">+1.51</span>
      </div>
      <div class="coef-row">
        <span class="coef-name">Obesity 3 yrs ago</span>
        <div class="coef-bar-wrap"><div class="coef-bar" style="width:22%; background:#5DCAA5;"></div></div>
        <span class="coef-val">+0.87</span>
      </div>
      <p style="font-size: 11px; color: #b4b2a9; margin-top: 6px;">Values on z-scored features &mdash; higher = stronger influence</p>
    </div>

    <div class="card">
      <p class="section-label">Global benchmarks</p>
      <p style="font-size: 12px; color: #888780; margin-bottom: 14px;">Reference points from WHO dataset</p>
      <div class="divider-row">
        <span>Global avg diabetes</span><span>7.9%</span>
      </div>
      <div class="divider-row">
        <span>Global avg obesity</span><span>14.9%</span>
      </div>
      <div class="divider-row">
        <span>Highest diabetes (Pacific)</span>
        <span style="color: #a32d2d;">~30%</span>
      </div>
      <div class="divider-row">
        <span>Lag window used</span><span>1, 2, 3 years</span>
      </div>
      <div class="divider-row">
        <span>Long-lag R&sup2; (5/7/10yr)</span>
        <span style="color: #854f0b;">0.693 &darr;</span>
      </div>
    </div>

  </div>

  <div class="card">
    <p class="section-label">Make a prediction</p>
    <p style="font-size: 13px; color: #5f5e5a; margin-bottom: 1rem;">
      Enter obesity prevalence values (%) for a country to predict its diabetes rate.
      Calls <code>POST /predict</code> on the local API.
    </p>

    <div class="slider-row">
      <label>Current obesity %</label>
      <input type="range" min="1" max="60" step="0.5" value="28.5" id="s0" oninput="sync(0,this.value)">
      <span class="val" id="v0">28.5</span>
    </div>
    <div class="slider-row">
      <label>Obesity 1 yr ago %</label>
      <input type="range" min="1" max="60" step="0.5" value="27.1" id="s1" oninput="sync(1,this.value)">
      <span class="val" id="v1">27.1</span>
    </div>
    <div class="slider-row">
      <label>Obesity 2 yrs ago %</label>
      <input type="range" min="1" max="60" step="0.5" value="25.8" id="s2" oninput="sync(2,this.value)">
      <span class="val" id="v2">25.8</span>
    </div>
    <div class="slider-row">
      <label>Obesity 3 yrs ago %</label>
      <input type="range" min="1" max="60" step="0.5" value="24.3" id="s3" oninput="sync(3,this.value)">
      <span class="val" id="v3">24.3</span>
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
      <div class="range-labels">
        <span>0%</span><span>Global avg 7.9%</span><span>30%</span>
      </div>
      <p style="font-size:12px; color:#888780; margin-top:10px;" id="result-note"></p>
    </div>

    <div class="error-box" id="error-box">
      <p id="error-msg"></p>
    </div>
  </div>

  <div class="info-box">
    <p>
      <i class="ti ti-info-circle" aria-hidden="true" style="font-size:14px; vertical-align:-2px; margin-right:4px;"></i>
      This dashboard calls your local API at <code>http://127.0.0.1:8000/predict</code>.
      Make sure <code>python src/serve_model.py</code> is running before making predictions.
    </p>
  </div>

  <script>
    const vals = [28.5, 27.1, 25.8, 24.3];

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

    async function runPredict() {
      document.getElementById('error-box').style.display = 'none';
      document.getElementById('result-box').style.display = 'none';

      const body = {
        obesity_current: vals[0],
        obesity_lag_1y:  vals[1],
        obesity_lag_2y:  vals[2],
        obesity_lag_3y:  vals[3]
      };

      try {
        const res = await fetch('http://127.0.0.1:8000/predict', {
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
          Math.abs(diff) + '% ' + dir + ' the global average of 7.9%. ' + data.note;

        document.getElementById('result-box').style.display = 'block';

      } catch (e) {
        document.getElementById('error-msg').textContent =
          'Could not reach the API: ' + e.message + '. Is serve_model.py running?';
        document.getElementById('error-box').style.display = 'block';
      }
    }
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