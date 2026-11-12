"""
Indian summer monsoon moisture trend, 1940 to present, from ERA5 reanalysis.

Run once:
    .venv\\Scripts\\python.exe monsoon_moisture.py

What it does
------------
1. Downloads ERA5 monthly means (June-September only) from the Copernicus
   Climate Data Store into ./data (skipped on later runs if already there):
     - specific humidity at 850 hPa
     - total column water vapour
     - land-sea mask and surface height (to pick the right grid points)
2. Averages each variable over Indian land, 8-28N, 68-90E, area-weighted,
   leaving out high ground (> 1500 m) where the 850 hPa level is underground.
3. Averages June-September for each year (the monsoon season, "JJAS").
4. Fits a straight-line trend per variable and tests if it is significant.
5. Saves:
     monsoon_moisture_trend.png   - two-panel chart
     monsoon_moisture_yearly.csv  - one row per year

Choices worth knowing for your report
-------------------------------------
- ERA5 before 1979 had no satellite observations and is less reliable; the
  chart shades that period. Trends are fitted over the whole period.
- Each grid cell is weighted by cos(latitude), its real area.
- Months are weighted equally inside the season.
"""

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

# ---------------------------------------------------------------- settings
FIRST_YEAR = 1940
MONTHS = ["06", "07", "08", "09"]          # June-September (JJAS)
NORTH, WEST, SOUTH, EAST = 28, 68, 8, 90   # region box, degrees
MAX_SURFACE_HEIGHT_M = 1500                # drop mountains above the 850 hPa level
SATELLITE_ERA_START = 1979
SMOOTHING_YEARS = 10

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CHART_FILE = HERE / "monsoon_moisture_trend.png"
CSV_FILE = HERE / "monsoon_moisture_yearly.csv"


def last_complete_season(today: date) -> int:
    """Latest year whose September is already in ERA5 (about a week's delay)."""
    return today.year if today.month >= 10 else today.year - 1


# ---------------------------------------------------------------- download
def _request(variable: str, years: list, months: list, **extra) -> dict:
    req = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [variable],
        "year": [str(y) for y in years],
        "month": months,
        "time": ["00:00"],
        "area": [NORTH, WEST, SOUTH, EAST],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    req.update(extra)
    return req


def download_all(data_dir: Path, last_year: int) -> None:
    """Fetch everything needed from Copernicus, skipping files already present."""
    data_dir.mkdir(exist_ok=True)
    years = list(range(FIRST_YEAR, last_year + 1))

    jobs = [
        ("q850.nc", "reanalysis-era5-pressure-levels-monthly-means",
         _request("specific_humidity", years, MONTHS, pressure_level=["850"])),
        ("tcwv.nc", "reanalysis-era5-single-levels-monthly-means",
         _request("total_column_water_vapour", years, MONTHS)),
        # invariant fields: one month is enough
        ("lsm.nc", "reanalysis-era5-single-levels-monthly-means",
         _request("land_sea_mask", [2000], ["06"])),
        ("orog.nc", "reanalysis-era5-single-levels-monthly-means",
         _request("geopotential", [2000], ["06"])),
    ]

    todo = [j for j in jobs if not (data_dir / j[0]).exists()]
    if not todo:
        print("All data already downloaded.")
        return

    try:
        import cdsapi
        client = cdsapi.Client()
    except Exception as err:  # most often: missing ~/.cdsapirc
        raise SystemExit(
            "\nCould not connect to Copernicus: %s\n\n"
            "Set up your key first:\n"
            "  1. Register at https://cds.climate.copernicus.eu and log in.\n"
            "  2. Accept the licence on both ERA5 monthly-means dataset pages\n"
            "     (Download tab -> 'Terms of use').\n"
            "  3. Copy your API key from your profile page and create the file\n"
            "     %s containing two lines:\n"
            "        url: https://cds.climate.copernicus.eu/api\n"
            "        key: <your-key>\n" % (err, Path.home() / ".cdsapirc")
        )

    for filename, dataset, request in todo:
        target = data_dir / filename
        print(f"Downloading {filename} (this can sit in the Copernicus queue for a few minutes)...")
        tmp = target.with_suffix(".part")
        client.retrieve(dataset, request).download(str(tmp))
        tmp.replace(target)  # only keep complete downloads


# ---------------------------------------------------------------- processing
def _open(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path)
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    # recent months can come split between final and preliminary ERA5
    if "expver" in ds.dims:
        ds = ds.mean("expver", skipna=True)
    for extra in ("pressure_level", "level"):
        if extra in ds.dims:
            ds = ds.isel({extra: 0}, drop=True)
    return ds


def _single_var(ds: xr.Dataset) -> xr.DataArray:
    data_vars = [v for v in ds.data_vars if v not in ("expver", "number")]
    return ds[data_vars[0]]


def region_mask(data_dir: Path) -> xr.DataArray:
    """True where the grid cell is mostly land and below MAX_SURFACE_HEIGHT_M."""
    lsm = _single_var(_open(data_dir / "lsm.nc")).squeeze(drop=True)
    height = _single_var(_open(data_dir / "orog.nc")).squeeze(drop=True) / 9.80665
    return (lsm > 0.5) & (height < MAX_SURFACE_HEIGHT_M)


def regional_jjas_mean(field: xr.DataArray, mask: xr.DataArray) -> pd.Series:
    """Area-weighted mean over the masked region, then June-September mean per year."""
    weights = np.cos(np.deg2rad(field.latitude)) * mask
    regional = field.weighted(weights.fillna(0)).mean(("latitude", "longitude"))
    regional = regional.sel(time=regional.time.dt.month.isin([6, 7, 8, 9]))
    yearly = regional.groupby("time.year").mean()
    return yearly.to_series()


def build_table(data_dir: Path) -> pd.DataFrame:
    mask = region_mask(data_dir)
    q = _single_var(_open(data_dir / "q850.nc")) * 1000.0      # kg/kg -> g/kg
    tcwv = _single_var(_open(data_dir / "tcwv.nc"))            # kg/m2
    mask = mask.reindex_like(q.isel(time=0, drop=True), method="nearest")
    table = pd.DataFrame({
        "q850_g_per_kg": regional_jjas_mean(q, mask),
        "tcwv_kg_per_m2": regional_jjas_mean(tcwv, mask),
    })
    table.index.name = "year"
    return table


# ---------------------------------------------------------------- statistics
def trend(series: pd.Series) -> dict:
    s = series.dropna()
    fit = stats.linregress(s.index.values, s.values)
    return {
        "slope_per_decade": fit.slope * 10,
        "intercept": fit.intercept,
        "slope": fit.slope,
        "p_value": fit.pvalue,
        "first": int(s.index.min()),
        "last": int(s.index.max()),
    }


# ---------------------------------------------------------------- chart
def plot(table: pd.DataFrame, trends: dict, out_file: Path) -> None:
    panels = [
        ("q850_g_per_kg", "Specific humidity at 850 hPa", "g/kg"),
        ("tcwv_kg_per_m2", "Total column water vapour", "kg/m²"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    years = table.index.values

    for ax, (col, title, unit) in zip(axes, panels):
        s = table[col]
        t = trends[col]
        #ax.axvspan(years.min() - 0.5, SATELLITE_ERA_START - 0.5, color="0.85", zorder=0,
        #           label="Before satellites (less reliable)")
        ax.plot(years, s.values, "-o", color="#7aa7d8", lw=1, ms=3, label="Jun-Aug mean, each year")
        smooth = s.rolling(SMOOTHING_YEARS, center=True).mean()
        ax.plot(years, smooth.values, color="#1f4e8c", lw=2.5, label=f"{SMOOTHING_YEARS}-year running mean")
        line = t["intercept"] + t["slope"] * years
        sig = "significant" if t["p_value"] < 0.05 else "not significant"
        ax.plot(years, line, "--", color="#d0452f", lw=2,
                label=f"Trend {t['slope_per_decade']:+.3f} {unit}/decade (p = {t['p_value']:.3g}, {sig})")
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    axes[-1].set_xlabel("Year")
    fig.suptitle(
        f"Indian summer monsoon moisture, June–September, {years.min()}–{years.max()}\n"
        f"ERA5 reanalysis, Indian land {SOUTH}–{NORTH}°N, {WEST}–{EAST}°E, below {MAX_SURFACE_HEIGHT_M} m",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    last_year = last_complete_season(date.today())
    download_all(DATA_DIR, last_year)

    table = build_table(DATA_DIR)
    table.round(4).to_csv(CSV_FILE)

    trends = {col: trend(table[col]) for col in table.columns}
    plot(table, trends, CHART_FILE)

    print(f"\nMonsoon season (JJAS) trends, {table.index.min()}-{table.index.max()}:")
    for col, unit in (("q850_g_per_kg", "g/kg"), ("tcwv_kg_per_m2", "kg/m2")):
        t = trends[col]
        verdict = "statistically significant" if t["p_value"] < 0.05 else "not statistically significant"
        print(f"  {col:16s} {t['slope_per_decade']:+.3f} {unit} per decade, "
              f"p = {t['p_value']:.3g} ({verdict} at 5%)")
    print(f"\nSaved {CHART_FILE.name} and {CSV_FILE.name}")


if __name__ == "__main__":
    main()
