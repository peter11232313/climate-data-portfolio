# Front Detection from Weather-Station Data

These scripts detect warm and cold front passages in 5-minute automatic weather-station records: pressure, dew point, relative humidity, wind speed and direction, and cumulative rain.

## Scripts

| Script | What it does |
|---|---|
| `front_passage_detection.py` | **After the event.** Compares the 60 minutes before and after every time step. It scores warm- and cold-front signatures (wind veer, dew-point step, change in pressure tendency, humidity, wind speed, rain onset) with weighted, capped indicators. It reports the top candidates, keeping them at least 3 hours apart, and plots them over the raw data. |
| `warm_front_nowcast.py` | **Real time (uses only past data).** A three-state machine, `NO SIGNAL → BEFORE → PASSED`. It enters BEFORE when pressure is falling and at least 2 of these signs appear: warming, wind backing, moistening, rain. It moves to PASSED when the pressure fall stops and there is a passage sign (wind veer, temperature step, rain stopping). It handles false alarms with a timeout and writes indicator-vs-threshold plots and a CSV of states. |

## Data-handling notes

- **Outlier removal.** A rolling 3-hour median filter on pressure removes logger start-up values.
- **Cumulative gauge.** Rain per step is the difference of the cumulative total, with a fix for the midnight reset.
- **Circular statistics.** Wind direction is averaged with unit vectors (cos/sin), and changes are wrapped to ±180° so that veering and backing are measured correctly across north.
- **Configurable thresholds.** All thresholds are named constants at the top of each script, ready to tune for another site.

## Input

An Excel file (`data.xlsx` / `d_data.xlsx`) with the columns `TimeStamp, P, Td, RH, U, Dirn, Rain total`. Station data is not included.
