"""
Indian summer monsoon wind trends, 1940 to present, from ERA5 reanalysis.

Run once:
    .venv\\Scripts\\python.exe monsoon_winds.py

What it does
------------
1. Downloads ERA5 monthly means (June-September only) from the Copernicus
   Climate Data Store into ./data (skipped on later runs if already there):
     - u and v wind at 850 hPa  (~1.5 km up, low-level jet)
     - u and v wind at 200 hPa  (~12 km up, upper-level jet)
     - land-sea mask (only to draw coastlines on the maps)
   Region downloaded: 0-30N, 40-100E.
2. Builds two jet-strength indices, area-weighted box averages:
     - Low-level jet (Somali jet / monsoon westerlies):
         850 hPa westerly wind (u), 5-15N, 50-70E
     - Tropical easterly jet:
         200 hPa easterly wind (-u), 5-15N, 40-80E
   For both, a bigger number means a stronger jet.
3. Fits a straight-line trend for June, July, August, September and the
   June-September season (JJAS), and tests if it is significant.
4. Saves:
     monsoon_winds_timeseries.png      - JJAS jet strength each year + trends
     monsoon_winds_monthly_trends.png  - trend per month and season, both jets
     monsoon_winds_trend_maps.png      - where the JJAS wind speed changed
     monsoon_winds_yearly.csv          - one row per year

Choices worth knowing for your report
-------------------------------------
- ERA5 before 1979 had no satellite observations and is less reliable,
  especially for upper-level winds. Charts shade that period, and the
  summary also gives the 1979-present trend for comparison.
- The jet indices use the east-west wind component, the usual way these
  jets are measured. Total wind speed in each box is also saved in the CSV.
- Each grid cell is weighted by cos(latitude), its real area.
- Months are weighted equally inside the season.
- Trend significance: two-sided t-test on the regression slope, 5% level.
"""

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Patch, Rectangle
from scipy import stats

# ---------------------------------------------------------------- settings
FIRST_YEAR = 1940
MONTHS = ["06", "07", "08", "09"]               # June-September (JJAS)
MONTH_NUMBERS = [6, 7, 8, 9]
NORTH, WEST, SOUTH, EAST = 30, 40, 0, 100       # download area, degrees
SATELLITE_ERA_START = 1979
SMOOTHING_YEARS = 10

# jet boxes: (south, north, west, east)
JETS = {
    "llj": {
        "name": "Low-level jet (850 hPa westerlies)",
        "short": "Low-level jet",
        "file": "wind850.nc",
        "box": (5, 15, 50, 70),
        "sign": +1,   # westerly wind is positive u
        "color": "#2a78d6",
    },
    "tej": {
        "name": "Tropical easterly jet (200 hPa easterlies)",
        "short": "Upper-level easterly jet",
        "file": "wind200.nc",
        "box": (5, 15, 40, 80),
        "sign": -1,   # easterly wind is negative u, flip so stronger = bigger
        "color": "#eb6834",
    },
}

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
TIMESERIES_FILE = HERE / "monsoon_winds_timeseries.png"
MONTHLY_FILE = HERE / "monsoon_winds_monthly_trends.png"
MAPS_FILE = HERE / "monsoon_winds_trend_maps.png"
CSV_FILE = HERE / "monsoon_winds_yearly.csv"

INK = "#0b0b0b"
INK_2 = "#52514e"


def last_complete_season(today: date) -> int:
    """Latest year whose September is already in ERA5 (about a week's delay)."""
    return today.year if today.month >= 10 else today.year - 1


# ---------------------------------------------------------------- download
def _request(variables: list, years: list, months: list, **extra) -> dict:
    req = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": variables,
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
    wind = ["u_component_of_wind", "v_component_of_wind"]

    jobs = [
        ("wind850.nc", "reanalysis-era5-pressure-levels-monthly-means",
         _request(wind, years, MONTHS, pressure_level=["850"])),
        ("wind200.nc", "reanalysis-era5-pressure-levels-monthly-means",
         _request(wind, years, MONTHS, pressure_level=["200"])),
        # invariant field: one month is enough
        ("lsm_winds.nc", "reanalysis-era5-single-levels-monthly-means",
         _request(["land_sea_mask"], [2000], ["06"])),
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
        print(f"Downloading {filename} (this can sit in the Copernicus queue for a while)...")
        tmp = target.with_suffix(".part")
        client.retrieve(dataset, request).download(str(tmp))
        tmp.replace(target)  # only keep complete downloads


# ---------------------------------------------------------------- loading
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
    drop = [c for c in ds.coords if c not in ds.dims]
    ds = ds.drop_vars(drop)
    if "latitude" in ds.dims:
        ds = ds.sortby("latitude")  # ERA5 comes north-to-south
    return ds


def load_wind(data_dir: Path, filename: str) -> xr.Dataset:
    ds = _open(data_dir / filename)[["u", "v"]]
    return ds.sel(time=ds.time.dt.month.isin(MONTH_NUMBERS)).load()


# ---------------------------------------------------------------- indices
def box_mean(field: xr.DataArray, box: tuple) -> xr.DataArray:
    south, north, west, east = box
    sub = field.sel(latitude=slice(south, north), longitude=slice(west, east))
    weights = np.cos(np.deg2rad(sub.latitude))
    return sub.weighted(weights).mean(("latitude", "longitude"))


def per_month_and_season(series: xr.DataArray, prefix: str) -> pd.DataFrame:
    """One column per month (Jun..Sep) plus the JJAS mean, indexed by year."""
    out = {}
    for m, label in zip(MONTH_NUMBERS, ["jun", "jul", "aug", "sep"]):
        sel = series.sel(time=series.time.dt.month == m)
        out[f"{prefix}_{label}"] = pd.Series(sel.values, index=sel.time.dt.year.values)
    jjas = series.groupby("time.year").mean()
    out[f"{prefix}_jjas"] = jjas.to_series()
    return pd.DataFrame(out)


def build_table(winds: dict) -> pd.DataFrame:
    parts = []
    for key, jet in JETS.items():
        ds = winds[key]
        strength = jet["sign"] * box_mean(ds.u, jet["box"])
        speed = box_mean(np.hypot(ds.u, ds.v), jet["box"])
        parts.append(per_month_and_season(strength, f"{key}_strength"))
        parts.append(per_month_and_season(speed, f"{key}_speed"))
    table = pd.concat(parts, axis=1)
    table.index.name = "year"
    # a season is only complete with all four months
    counts = winds["llj"].u.time.dt.year.to_series().value_counts()
    full_years = counts[counts == 4].index
    table.loc[~table.index.isin(full_years), table.columns.str.endswith("_jjas")] = np.nan
    return table


# ---------------------------------------------------------------- statistics
def trend(series: pd.Series, first_year: int | None = None) -> dict:
    s = series.dropna()
    if first_year is not None:
        s = s[s.index >= first_year]
    fit = stats.linregress(s.index.values, s.values)
    t_crit = stats.t.ppf(0.975, len(s) - 2)
    return {
        "slope_per_decade": fit.slope * 10,
        "ci95_per_decade": t_crit * fit.stderr * 10,
        "intercept": fit.intercept,
        "slope": fit.slope,
        "p_value": fit.pvalue,
        "first": int(s.index.min()),
        "last": int(s.index.max()),
    }


def gridded_trend(yearly: xr.DataArray) -> tuple:
    """Least-squares slope (per decade) and p-value at every grid point."""
    years = yearly.year.values.astype(float)
    x = years - years.mean()
    y = yearly.values
    y_anom = y - y.mean(axis=0)
    sxx = (x ** 2).sum()
    slope = np.tensordot(x, y_anom, axes=(0, 0)) / sxx
    resid = y_anom - x[:, None, None] * slope
    n = len(x)
    se = np.sqrt((resid ** 2).sum(axis=0) / (n - 2) / sxx)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = slope / se
    p = 2 * stats.t.sf(np.abs(t), n - 2)
    coords = {"latitude": yearly.latitude, "longitude": yearly.longitude}
    dims = ("latitude", "longitude")
    return (xr.DataArray(slope * 10, coords=coords, dims=dims),
            xr.DataArray(p, coords=coords, dims=dims))


# ---------------------------------------------------------------- charts
def _style(ax) -> None:
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(INK_2)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def _sig_word(p: float) -> str:
    return "significant" if p < 0.05 else "not significant"


def plot_timeseries(table: pd.DataFrame, full: dict, sat: dict, out_file: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, (key, jet) in zip(axes, JETS.items()):
        col = f"{key}_strength_jjas"
        s = table[col].dropna()
        years = s.index.values
        ax.axvspan(years.min() - 0.5, SATELLITE_ERA_START - 0.5, color="0.9", zorder=0,
                   label="Before satellites (less reliable)")
        ax.plot(years, s.values, "-o", color=jet["color"], alpha=0.45, lw=1, ms=3,
                label="JJAS mean, each year")
        smooth = s.rolling(SMOOTHING_YEARS, center=True).mean()
        ax.plot(years, smooth.values, color=jet["color"], lw=2.5,
                label=f"{SMOOTHING_YEARS}-year running mean")
        for t, style, label in ((full[col], "--", "Trend"), (sat[col], ":", "Trend since 1979")):
            yrs = np.arange(t["first"], t["last"] + 1)
            ax.plot(yrs, t["intercept"] + t["slope"] * yrs, style, color=INK, lw=2,
                    label=f"{label} {t['slope_per_decade']:+.2f} m/s per decade "
                          f"(p = {t['p_value']:.2g}, {_sig_word(t['p_value'])})")
        south, north, west, east = jet["box"]
        ax.set_title(f"{jet['name']}, {south}–{north}°N, {west}–{east}°E",
                     loc="left", fontsize=12, color=INK)
        ax.set_ylabel("Jet strength (m/s)", color=INK_2)
        _style(ax)
        ax.legend(fontsize=8, loc="best", frameon=False)
    axes[-1].set_xlabel("Year", color=INK_2)
    first, last = int(table.index.min()), int(table[f"llj_strength_jjas"].dropna().index.max())
    fig.suptitle(f"Indian summer monsoon jets, June–September, {first}–{last}\n"
                 "ERA5 reanalysis. Higher = stronger jet.", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def plot_monthly_trends(full: dict, out_file: Path) -> None:
    periods = [("jun", "June"), ("jul", "July"), ("aug", "August"),
               ("sep", "September"), ("jjas", "June–Sept\nseason")]
    x = np.arange(len(periods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (key, jet) in enumerate(JETS.items()):
        offs = (i - 0.5) * width
        for j, (code, _) in enumerate(periods):
            t = full[f"{key}_strength_{code}"]
            sig = t["p_value"] < 0.05
            ax.bar(x[j] + offs, t["slope_per_decade"], width * 0.92,
                   color=jet["color"] if sig else "white",
                   edgecolor=jet["color"], linewidth=1.8,
                   hatch=None if sig else "///", zorder=3)
            ax.errorbar(x[j] + offs, t["slope_per_decade"], yerr=t["ci95_per_decade"],
                        color=INK_2, lw=1, capsize=3, zorder=4)
            va = "bottom" if t["slope_per_decade"] >= 0 else "top"
            pad = t["ci95_per_decade"] + 0.01
            ax.text(x[j] + offs, t["slope_per_decade"] + (pad if va == "bottom" else -pad),
                    f"{t['slope_per_decade']:+.2f}", ha="center", va=va, fontsize=8, color=INK)
    ax.axhline(0, color=INK_2, lw=1)
    ax.axvline(3.5, color="0.8", lw=1, ls="-")
    ax.set_xticks(x)
    ax.set_xticklabels([p[1] for p in periods], color=INK)
    ax.set_ylabel("Change in jet strength (m/s per decade)", color=INK_2)
    _style(ax)
    ax.grid(axis="x", visible=False)
    handles = [
        Patch(facecolor=JETS["llj"]["color"], edgecolor=JETS["llj"]["color"], label=JETS["llj"]["name"]),
        Patch(facecolor=JETS["tej"]["color"], edgecolor=JETS["tej"]["color"], label=JETS["tej"]["name"]),
        Patch(facecolor="white", edgecolor=INK_2, hatch="///", label="Hatched = not significant (5% level)"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="best")
    t0 = full["llj_strength_jjas"]
    ax.set_title(f"Monsoon jet trends by month, {t0['first']}–{t0['last']} (ERA5)\n"
                 "Above zero = jet getting stronger. Whiskers = 95% confidence range.",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def plot_trend_maps(winds: dict, lsm: xr.DataArray | None, out_file: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9.5))
    levels = np.linspace(-0.6, 0.6, 13)
    for ax, (key, jet) in zip(axes, JETS.items()):
        ds = winds[key]
        counts = ds.time.dt.year.to_series().value_counts()
        full_years = counts[counts == 4].index.values
        ds = ds.sel(time=ds.time.dt.year.isin(full_years))
        yearly = ds.groupby("time.year").mean()
        speed = np.hypot(yearly.u, yearly.v)
        slope, p = gridded_trend(speed)

        cf = ax.contourf(slope.longitude, slope.latitude, slope.clip(levels[0], levels[-1]),
                         levels=levels, cmap="RdBu_r", extend="both")
        # stipple where not significant would clutter; dot where significant
        sig = p < 0.05
        step = 8
        lon2, lat2 = np.meshgrid(slope.longitude.values[::step], slope.latitude.values[::step])
        ax.scatter(lon2[sig.values[::step, ::step]], lat2[sig.values[::step, ::step]],
                   s=1.5, color=INK, alpha=0.6, linewidths=0)
        # mean wind arrows: where the jet is
        mean_u = yearly.u.mean("year")
        mean_v = yearly.v.mean("year")
        q_step = 12
        q = ax.quiver(mean_u.longitude[::q_step], mean_u.latitude[::q_step],
                      mean_u.values[::q_step, ::q_step], mean_v.values[::q_step, ::q_step],
                      color="0.25", scale=400 if key == "tej" else 250, width=0.0022)
        ax.quiverkey(q, 0.93, 1.03, 10, "10 m/s mean wind", labelpos="W",
                     fontproperties={"size": 8})
        if lsm is not None:  # coastlines, only if data/lsm_winds.nc exists
            ax.contour(lsm.longitude, lsm.latitude, lsm, levels=[0.5],
                       colors=INK, linewidths=0.8)
        south, north, west, east = jet["box"]
        ax.add_patch(Rectangle((west, south), east - west, north - south, fill=False,
                               edgecolor=jet["color"], lw=2.2, zorder=5))
        ax.set_aspect("equal")
        ax.set_xlim(WEST, EAST)
        ax.set_ylim(SOUTH, NORTH)
        ax.set_xlabel("Longitude (°E)", color=INK_2, fontsize=9)
        ax.set_ylabel("Latitude (°N)", color=INK_2, fontsize=9)
        ax.tick_params(colors=INK_2, labelsize=8)
        level = "850 hPa (~1.5 km)" if key == "llj" else "200 hPa (~12 km)"
        ax.set_title(f"{level}: trend in June–Sept wind speed, "
                     f"{int(full_years.min())}–{int(full_years.max())}",
                     loc="left", fontsize=11, color=INK)
        cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("m/s per decade", color=INK_2, fontsize=9)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle("Where monsoon winds changed (ERA5). Red = faster, blue = slower.\n"
                 "Arrows: average wind. Dots: trend significant at 5%. "
                 "Coloured box: jet index region.", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:

    winds = {key: load_wind(DATA_DIR, jet["file"]) for key, jet in JETS.items()}

    table = build_table(winds)
    table.round(3).to_csv(CSV_FILE)

    full = {col: trend(table[col]) for col in table.columns}
    sat = {col: trend(table[col], SATELLITE_ERA_START) for col in table.columns}

    plot_timeseries(table, full, sat, TIMESERIES_FILE)
    plot_monthly_trends(full, MONTHLY_FILE)

    lsm_file = DATA_DIR / "lsm_winds.nc"
    lsm = _open(lsm_file)["lsm"].squeeze(drop=True) if lsm_file.exists() else None
    if lsm is None:
        print("No data/lsm_winds.nc found: drawing maps without coastlines.")
    plot_trend_maps(winds, lsm, MAPS_FILE)

    print("\nJet strength trends, m/s per decade (positive = stronger jet):")
    header = f"  {'':26s}{'Jun':>9s}{'Jul':>9s}{'Aug':>9s}{'Sep':>9s}{'JJAS':>9s}"
    for label, results in (("whole record", full), ("since 1979", sat)):
        t0 = results["llj_strength_jjas"]
        print(f"\n {label} ({t0['first']}-{t0['last']}); * = significant at 5%")
        print(header)
        for key, jet in JETS.items():
            cells = ""
            for code in ("jun", "jul", "aug", "sep", "jjas"):
                t = results[f"{key}_strength_{code}"]
                star = "*" if t["p_value"] < 0.05 else " "
                cells += f"{t['slope_per_decade']:+8.2f}{star}"
            print(f"  {jet['short']:26s}{cells}")
    print(f"\nSaved {TIMESERIES_FILE.name}, {MONTHLY_FILE.name}, {MAPS_FILE.name}, {CSV_FILE.name}")


if __name__ == "__main__":
    main()
