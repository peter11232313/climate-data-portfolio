# Climate & Weather Data Portfolio

Python work from my MSc in Atmosphere, Ocean and Climate (University of Reading).
It covers reanalysis data engineering, numerical methods and observational data analysis.

| Project | What it shows | Stack |
|---|---|---|
| [`era5-monsoon-trends/`](era5-monsoon-trends/) | A reproducible ERA5 pipeline: downloads from the Copernicus CDS API, area-weighted climate indices, trend tests at every grid point, publication-quality maps and charts | xarray, pandas, NumPy, SciPy, Matplotlib, cdsapi |
| [`numerical-advection/`](numerical-advection/) | A TVD advection scheme with a van Leer flux limiter, plus a test harness that compares scheme boundedness across Courant numbers | NumPy, Matplotlib |
| [`weather-station-fronts/`](weather-station-fronts/) | Detecting warm and cold front passages in 5-minute weather-station data: signal cleaning, circular statistics for wind direction, a scoring model and a real-time state machine | pandas, NumPy, Matplotlib |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

See each project's README for its data and how to run it.
