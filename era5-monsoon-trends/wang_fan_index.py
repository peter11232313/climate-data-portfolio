"""
Wang-Fan Indian summer monsoon circulation index, 1940 to present, from ERA5.

Run (after monsoon_winds.py has downloaded the wind data):
    .venv\\Scripts\\python.exe wang_fan_index.py

Definition (Wang and Fan 1999, Bull. Amer. Meteor. Soc., 80, 629-638,
"MCI1"/"WSI1", their recommended ISM circulation index):

    index = U850 - U200, averaged over 5-20N, 40-80E

U is the east-west wind (westerly positive). In summer the monsoon has
westerlies at 850 hPa and easterlies at 200 hPa, so a larger index means a
stronger vertical wind shear, i.e. a stronger monsoon circulation.
Wang and Fan used June-September monthly means; here the four months are
averaged into one value per year.

Inputs (already downloaded by monsoon_winds.py):
    data/wind850.nc, data/wind200.nc   (ERA5 monthly means, Jun-Sep)

Saves:
    wang_fan_index.png   - yearly index, running mean, trends
    wang_fan_index.csv   - one row per year
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from monsoon_winds import (DATA_DIR, INK, INK_2, SATELLITE_ERA_START, SMOOTHING_YEARS,
                           _style, box_mean, load_wind, trend)

BOX = (5, 20, 40, 80)  # south, north, west, east (Wang and Fan 1999)
COLOR = "#2a78d6"

HERE = Path(__file__).resolve().parent
CHART_FILE = HERE / "wang_fan_index.png"
CSV_FILE = HERE / "wang_fan_index.csv"


def wang_fan_index() -> pd.Series:
    """June-September mean of U850 - U200 over the Wang-Fan box, one value per year."""
    u850 = load_wind(DATA_DIR, "wind850.nc").u
    u200 = load_wind(DATA_DIR, "wind200.nc").u
    monthly = box_mean(u850 - u200, BOX)          # area-weighted, one value per month
    df = monthly.to_series().to_frame("index")
    df["year"] = df.index.year
    grouped = df.groupby("year")["index"]
    yearly = grouped.mean()
    yearly[grouped.count() < 4] = float("nan")    # only complete Jun-Sep seasons
    yearly.index.name = "year"
    return yearly.dropna()


def sig_word(p: float) -> str:
    return "significant" if p < 0.05 else "not significant"


def plot(series: pd.Series, full: dict, sat: dict) -> None:
    years = series.index.values
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(years.min() - 0.5, SATELLITE_ERA_START - 0.5, color="0.9", zorder=0,
               label="Before satellites (less reliable)")
    ax.plot(years, series.values, "-o", color=COLOR, alpha=0.45, lw=1, ms=3,
            label="June–September mean, each year")
    smooth = series.rolling(SMOOTHING_YEARS, center=True).mean()
    ax.plot(years, smooth.values, color=COLOR, lw=2.5,
            label=f"{SMOOTHING_YEARS}-year running mean")
    for t, style, label in ((full, "--", "Trend"), (sat, ":", f"Trend since {SATELLITE_ERA_START}")):
        yrs = range(t["first"], t["last"] + 1)
        ax.plot(list(yrs), [t["intercept"] + t["slope"] * y for y in yrs], style, color=INK, lw=2,
                label=f"{label} {t['slope_per_decade']:+.2f} m/s per decade "
                      f"(p = {t['p_value']:.2g}, {sig_word(t['p_value'])})")
    south, north, west, east = BOX
    ax.set_title(f"Wang–Fan monsoon circulation index (U850 − U200, {south}–{north}°N, "
                 f"{west}–{east}°E)\nJune–September, {years.min()}–{years.max()}, ERA5 reanalysis. "
                 "Higher = stronger monsoon circulation.",
                 loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Year", color=INK_2)
    ax.set_ylabel("U850 − U200 (m/s)", color=INK_2)
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(CHART_FILE, dpi=200)
    plt.close(fig)


def main() -> None:
    series = wang_fan_index()
    series.round(3).to_frame("wang_fan_index_m_per_s").to_csv(CSV_FILE)
    full = trend(series)
    sat = trend(series, SATELLITE_ERA_START)
    plot(series, full, sat)

    print("Wang-Fan index (U850 - U200, 5-20N, 40-80E), June-September mean")
    print(f"  mean {series.mean():.2f} m/s, years {series.index.min()}-{series.index.max()}")
    for label, t in (("whole record", full), (f"since {SATELLITE_ERA_START}", sat)):
        print(f"  {label:14s} {t['slope_per_decade']:+.3f} m/s per decade "
              f"(95% range +/-{t['ci95_per_decade']:.3f}), p = {t['p_value']:.3g} ({sig_word(t['p_value'])})")
    print(f"Saved {CHART_FILE.name}, {CSV_FILE.name}")


if __name__ == "__main__":
    main()
