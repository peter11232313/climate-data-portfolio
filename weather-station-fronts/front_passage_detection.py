"""
Front passage detection (warm AND cold) from 5-min weather-station data.
Columns used: TimeStamp, P (hPa), Td (deg C), RH (%), U (m/s), Dirn (deg, from), Rain total (mm)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ settings
FILE, SHEET = "d_data.xlsx", 0
WINDOW      = "60min"      # compare 60 min before vs 60 min after each time
SMOOTH      = "15min"
MIN_POINTS  = 6
MIN_SEP_H   = 3            # min separation between reported events (hours)
THR = dict(veer=30.0, temp=0.8, pres=0.5, rh=5.0, wspd=1.0)   # deg, degC, hPa/h, %, m/s
W   = dict(veer=2.0, temp=2.0, pres=1.5, rh=1.0, wspd=0.5, rain=1.0)

# ------------------------------------------------------------------ load & clean
df = pd.read_excel(FILE, sheet_name=SHEET)
df.columns = [str(c).strip() for c in df.columns]
df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], format="%Y/%m/%d %H:%M", errors="coerce")
df = df.dropna(subset=["TimeStamp"]).set_index("TimeStamp").sort_index()
df = df[~df.index.duplicated()]
cols = ["P", "Td", "RH", "U", "Dirn", "Rain total"]
df = df[cols].apply(pd.to_numeric, errors="coerce").dropna(subset=["P", "Td"])

# the very first record (12:45) is a start-up value (P = 960.8 hPa) -> remove outliers
med = df["P"].rolling("3h", min_periods=3, center=True).median()
df = df[(df["P"] - med).abs() < 5]

# rain rate (mm / 5 min) from the cumulative gauge, which resets at midnight
rain = df["Rain total"].diff()
rain[rain < 0] = df["Rain total"][rain < 0]          # after reset, the new value is the increment
df["Rain"] = rain.clip(lower=0).fillna(0)

# ------------------------------------------------------------------ smoothing
r = np.deg2rad(df["Dirn"])
cx = np.cos(r).rolling(SMOOTH, min_periods=1).mean()
cy = np.sin(r).rolling(SMOOTH, min_periods=1).mean()
sm = df[["P", "Td", "RH", "U"]].rolling(SMOOTH, min_periods=1).mean()
sm["Dirn"] = np.rad2deg(np.arctan2(cy, cx)) % 360
sm["Rain"] = df["Rain"]

def circ_mean(d):
    d = np.deg2rad(d.dropna()); return np.rad2deg(np.arctan2(np.sin(d).mean(), np.cos(d).mean())) % 360
def sdiff(a, b):                       # b - a, >0 = clockwise (veering)
    return (b - a + 180) % 360 - 180
def slope(s):                          # hPa per hour
    s = s.dropna()
    if len(s) < 3: return np.nan
    t = (s.index - s.index[0]).total_seconds() / 3600
    return np.polyfit(t, s.values, 1)[0]
clip = lambda x, t: float(np.clip(x / t, 0, 2))

# ------------------------------------------------------------------ scoring
w, rows = pd.Timedelta(WINDOW), []
for t in sm.index:
    b = sm.loc[t - w: t - pd.Timedelta("1s")]
    a = sm.loc[t: t + w]
    if b["Td"].count() < MIN_POINTS or a["Td"].count() < MIN_POINTS:
        continue
    veer = sdiff(circ_mean(b["Dirn"]), circ_mean(a["Dirn"]))
    dT   = a["Td"].mean() - b["Td"].mean()
    dRH  = a["RH"].mean() - b["RH"].mean()
    dU   = a["U"].mean()  - b["U"].mean()
    pb, pa = slope(b["P"]), slope(a["P"])
    dPt  = pa - pb
    rain_on = float(b["Rain"].sum() < 0.05 and a["Rain"].sum() >= 0.2)   # rain starts at t
    rain_any = float(a["Rain"].sum() >= 0.2)

    warm = (W["veer"]*clip(veer, THR["veer"]) + W["temp"]*clip(dT, THR["temp"])
            + W["pres"]*clip(dPt, THR["pres"])*(pb < 0) + W["rh"]*clip(dRH, THR["rh"])
            + W["wspd"]*clip(abs(dU), THR["wspd"]) + W["rain"]*rain_any)
    cold = (W["veer"]*clip(veer, THR["veer"]) + W["temp"]*clip(-dT, THR["temp"])
            + W["pres"]*clip(dPt, THR["pres"]) + W["rh"]*clip(abs(dRH), THR["rh"])
            + W["wspd"]*clip(abs(dU), THR["wspd"]) + W["rain"]*rain_on)
    rows.append(dict(time=t, warm=warm, cold=cold, veer=veer, dT=dT, dRH=dRH, dU=dU,
                     p_slope_before=pb, p_slope_after=pa, rain_starts=rain_on))
res = pd.DataFrame(rows).set_index("time")
if res.empty: raise SystemExit("Not enough valid data.")

def top_events(col, n=3):
    out, s = [], res[col].sort_values(ascending=False)
    for t, v in s.items():
        if all(abs((t - o).total_seconds()) > MIN_SEP_H * 3600 for o in out):
            out.append(t)
        if len(out) == n: break
    return out

maxs = sum(W.values()) * 2
for kind in ("warm", "cold"):
    print(f"\n=== {kind.upper()} front candidates ===")
    for t in top_events(kind):
        row = res.loc[t]
        print(f"{t:%m-%d %H:%M}  score {row[kind]/maxs:.0%} | veer {row.veer:+.0f} deg | dTd {row.dT:+.2f} | "
              f"dRH {row.dRH:+.1f} | dU {row.dU:+.1f} | dP/dt before {row.p_slope_before:+.2f} -> after {row.p_slope_after:+.2f} hPa/h")

best_warm, best_cold = res["warm"].idxmax(), res["cold"].idxmax()

# ------------------------------------------------------------------ plot
fig, ax = plt.subplots(6, 1, figsize=(12, 13), sharex=True)
for a_, c, lab in zip(ax[:4], ["P", "Td", "RH", "U"], ["P (hPa)", "Td (°C)", "RH (%)", "U (m/s)"]):
    a_.plot(df.index, df[c], ".", alpha=.3); a_.plot(sm.index, sm[c]); a_.set_ylabel(lab)
ax[4].plot(df.index, df["Dirn"], ".", alpha=.5); ax[4].set_ylim(0, 360)
ax[4].set_yticks(range(0, 361, 90)); ax[4].set_ylabel("Wind dir (°)")
ax[5].bar(df.index, df["Rain"], width=0.003); ax[5].set_ylabel("Rain (mm/5 min)")
for a_ in ax:
    a_.axvline(best_warm, color="red", ls="--", label="warm"); a_.axvline(best_cold, color="blue", ls="--", label="cold")
    a_.grid(alpha=.3)
ax[0].legend(); ax[0].set_title(f"Warm: {best_warm:%m-%d %H:%M}   Cold: {best_cold:%m-%d %H:%M}")
plt.tight_layout(); plt.savefig("front_analysis.png", dpi=150); plt.show()
res.to_csv("front_scores.csv")