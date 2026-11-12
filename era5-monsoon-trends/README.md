# Indian Summer Monsoon Trends from ERA5 (1940–present)

These scripts ask whether the Indian summer monsoon (June–September, "JJAS") has changed since 1940, using ERA5 reanalysis.

## Scripts

| Script | What it does |
|---|---|
| `monsoon_winds.py` | Downloads ERA5 monthly u/v winds at 850 and 200 hPa. Builds jet-strength indices for the **Somali low-level jet** (850 hPa westerlies, 5–15°N, 50–70°E) and the **Tropical Easterly Jet** (200 hPa easterlies, 5–15°N, 40–80°E). Fits trends per month and per season, and maps the trend in wind speed at every grid point, with significance stippling. |
| `wang_fan_index.py` | Computes the **Wang–Fan (1999) monsoon circulation index**, U850 − U200 over 5–20°N, 40–80°E, and its trend. It reuses the data and helpers from `monsoon_winds.py`. |
| `monsoon_moisture.py` | Trends in 850 hPa specific humidity and total column water vapour over Indian land (8–28°N, 68–90°E). It uses the ERA5 land–sea mask and leaves out high ground (> 1500 m), where the 850 hPa level is underground. |

## Engineering notes

- **Reproducible downloads.** Requests go through the Copernicus CDS API. Files that already exist are skipped, and each download goes to a `.part` file that is renamed only once it is complete.
- **Handles ERA5 data quirks.** The loader renames `valid_time` to `time`, merges the ERA5/ERA5T (`expver`) split for recent months, and sorts latitude from south to north.
- **Correct averaging.** Grid cells are weighted by cos(latitude). A season counts only if all four months are present.
- **Vectorised statistics.** Least-squares slope, standard error and two-sided t-test at every grid point in plain NumPy (`gridded_trend`), with 95% confidence intervals on index trends.
- **Honest about data quality.** ERA5 is less reliable before satellite data (pre-1979). Charts shade that period, and every trend is also reported for 1979 onwards.

## Run

1. Register at <https://cds.climate.copernicus.eu>, accept the ERA5 monthly-means licences, and put your API key in `~/.cdsapirc`.
2. Run:

```bash
python monsoon_winds.py      # downloads winds into ./data, then analyses them
python wang_fan_index.py     # needs the wind data from step above
python monsoon_moisture.py
```

Outputs (PNG charts and per-year CSVs) are written next to the scripts.
