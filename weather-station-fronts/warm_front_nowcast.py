"""
Is it still BEFORE the warm front passes?   (uses only past data -> can run in real time)

States
  NO SIGNAL : no sign of an approaching warm front
  BEFORE    : pre-frontal conditions (pressure falling + warming/backing wind/moister/rain)
  PASSED    : after BEFORE, the pressure fall stopped AND >=1 passage sign appeared
              (clockwise wind shift, Td step up, rain stopping)

Columns: TimeStamp, P (hPa), Td (deg C), RH (%), U (m/s), Dirn (deg, from), Rain total (mm)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE, SHEET = "data.xlsx", 0

# ---- thresholds (empirical starting values - tune for your site) -------------
P_FALL_3H   = -1.0   # hPa / 3 h   pressure falling at least this much  -> pre-frontal
T_RISE_3H   =  0.5   # degC / 3 h  warming
BACKING_3H  = -15.0  # deg / 3 h   wind backing (counter-clockwise)
RH_RISE_3H  =  5.0   # % / 3 h
RAIN_3H     =  0.2   # mm / 3 h    rain has started
MIN_EXTRA   =  2     # pre-frontal signs needed besides the pressure fall

P_FLAT_1H   = -0.1   # hPa / h     fall has stopped (>= this)
VEER_1H     =  30.0  # deg / h     clockwise shift  -> passage sign
T_STEP_1H   =  0.8   # degC / h    temperature step -> passage sign
LOST_AFTER_H = 3     # BEFORE -> NO SIGNAL if conditions vanish this long (false alarm)
AFTER_H      = 24    # PASSED -> NO SIGNAL after this many hours

# ------------------------------------------------------------------ load & clean
df = pd.read_excel(FILE, sheet_name=SHEET)
df.columns = [str(c).strip() for c in df.columns]
df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], format="%Y/%m/%d %H:%M", errors="coerce")
df = df.dropna(subset=["TimeStamp"]).set_index("TimeStamp").sort_index()
df = df[~df.index.duplicated()]
df = df[["P", "Td", "RH", "U", "Dirn", "Rain total"]].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=["P", "Td"])
med = df["P"].rolling("3h", min_periods=3, center=True).median()
df = df[(df["P"] - med).abs() < 5]                      # drop start-up outliers (e.g. 960 hPa)

# rain increments from cumulative gauge (resets at midnight)
rain = df["Rain total"].diff()
rain[rain < 0] = df["Rain total"][rain < 0]
df["Rain"] = rain.clip(lower=0).fillna(0)

# ------------------------------------------------------------------ regular 5-min grid
r = np.deg2rad(df["Dirn"])
df["cx"], df["cy"] = np.cos(r), np.sin(r)
g = df[["P", "Td", "RH", "U", "cx", "cy"]].resample("5min").mean()
g["Rain"] = df["Rain"].resample("5min").sum()
g = g.interpolate(limit=3)                              # fill tiny gaps only
sm = g[["P", "Td", "RH", "U", "cx", "cy"]].rolling(6, min_periods=3).mean()    # 30-min smoothing
sm["Dir"] = np.rad2deg(np.arctan2(sm["cy"], sm["cx"])) % 360
sm["Rain"] = g["Rain"]

n1, n3 = 12, 36                                         # 1 h and 3 h in 5-min steps
sdiff = lambda a, b: (a - b + 180) % 360 - 180          # a - b, >0 = clockwise (veer)

f = pd.DataFrame(index=sm.index)
f["dP3"]  = sm["P"]  - sm["P"].shift(n3)
f["dP1"]  = sm["P"]  - sm["P"].shift(n1)
f["dT3"]  = sm["Td"] - sm["Td"].shift(n3)
f["dT1"]  = sm["Td"] - sm["Td"].shift(n1)
f["dRH3"] = sm["RH"] - sm["RH"].shift(n3)
f["dDir3"] = sdiff(sm["Dir"], sm["Dir"].shift(n3))
f["dDir1"] = sdiff(sm["Dir"], sm["Dir"].shift(n1))
f["rain3"] = sm["Rain"].rolling(n3).sum()
f["rain1"] = sm["Rain"].rolling(n1).sum()
f["rain_prev2h"] = sm["Rain"].shift(n1).rolling(2 * n1).sum()

# ------------------------------------------------------------------ signs
pre_signs = pd.DataFrame({
    "warming": f["dT3"]   >= T_RISE_3H,
    "backing": f["dDir3"] <= BACKING_3H,
    "moister": f["dRH3"]  >= RH_RISE_3H,
    "rain":    f["rain3"] >= RAIN_3H,
})
f["pre_signs"] = pre_signs.sum(axis=1)
f["pre"] = (f["dP3"] <= P_FALL_3H) & (f["pre_signs"] >= MIN_EXTRA)

pass_signs = pd.DataFrame({
    "veer":      f["dDir1"] >= VEER_1H,
    "T_step":    f["dT1"]   >= T_STEP_1H,
    "rain_stop": (f["rain1"] < RAIN_3H) & (f["rain_prev2h"] >= 0.5),
})
f["pass_signs"] = pass_signs.sum(axis=1)
f["p_flat"] = f["dP1"] >= P_FLAT_1H
f["passage"] = f["p_flat"] & (f["pass_signs"] >= 1)

# ------------------------------------------------------------------ state machine
state, since, lost, states = "NO SIGNAL", None, 0, []
for t, row in f.iterrows():
    if state == "NO SIGNAL":
        if row["pre"]:
            state, since, lost = "BEFORE", t, 0
    elif state == "BEFORE":
        if row["passage"]:
            state, since = "PASSED", t
        else:
            lost = 0 if row["pre"] else lost + 1
            if lost * 5 / 60 >= LOST_AFTER_H:
                state, since = "NO SIGNAL", t
    elif state == "PASSED":
        if (t - since) > pd.Timedelta(hours=AFTER_H):
            state, since = "NO SIGNAL", t
    states.append(state)
f["state"] = states

# ------------------------------------------------------------------ report
changes = f["state"].ne(f["state"].shift())
print("State changes:")
for t, s in f.loc[changes, "state"].items():
    print(f"  {t:%Y-%m-%d %H:%M}  ->  {s}")

now = f.index[-1]
last = f.iloc[-1]
print(f"\nLatest record: {now:%Y-%m-%d %H:%M}   STATE = {last['state']}")
print(f"  P 3h change {last['dP3']:+.2f} hPa | Td 3h {last['dT3']:+.2f} | RH 3h {last['dRH3']:+.1f} "
      f"| wind dir 3h {last['dDir3']:+.0f} deg | rain 3h {last['rain3']:.2f} mm")
if last["state"] == "BEFORE":
    print("  -> Warm front has NOT passed yet (pre-frontal conditions).")
elif last["state"] == "PASSED":
    print("  -> Warm front has probably PASSED.")
else:
    print("  -> No clear warm-front signal.")

# ------------------------------------------------------------------ plot
colors = {"BEFORE": "orange", "PASSED": "green", "NO SIGNAL": None}
fig, ax = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
ax[0].plot(sm.index, sm["P"]);   ax[0].set_ylabel("P (hPa)")
ax[1].plot(sm.index, sm["Td"]);  ax[1].set_ylabel("Td (°C)")
ax[2].plot(sm.index, sm["RH"]);  ax[2].set_ylabel("RH (%)")
ax[3].plot(sm.index, sm["Dir"], ".", ms=2); ax[3].set_ylim(0, 360); ax[3].set_ylabel("Wind dir (°)")
ax[4].bar(sm.index, sm["Rain"], width=0.003); ax[4].set_ylabel("Rain (mm/5min)")
seg = (f["state"] != f["state"].shift()).cumsum()
for _, part in f.groupby(seg):
    c = colors[part["state"].iloc[0]]
    if c:
        for a in ax:
            a.axvspan(part.index[0], part.index[-1], color=c, alpha=.15)
ax[0].set_title("Orange = BEFORE warm front,  Green = PASSED")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("before_warm_front.png", dpi=150)

# ------------------------------------------------------------------ plot 2: indicators vs thresholds
def panel(a, series, label, thr=None, thr_label=None, below=False):
    a.plot(f.index, series, lw=1.2)
    if thr is not None:
        a.axhline(thr, color="red", ls="--", lw=1, label=thr_label)
        ok = series <= thr if below else series >= thr
        a.fill_between(f.index, series, thr, where=ok, color="red", alpha=.25)
        a.legend(loc="upper left", fontsize=8)
    a.set_ylabel(label); a.grid(alpha=.3)

fig2, ax2 = plt.subplots(8, 1, figsize=(12, 17), sharex=True)
panel(ax2[0], f["dP3"],   "P 3h change\n(hPa)",   P_FALL_3H,  f"fall <= {P_FALL_3H}", below=True)
panel(ax2[1], f["dT3"],   "Td 3h change\n(°C)",   T_RISE_3H,  f"rise >= {T_RISE_3H}")
panel(ax2[2], f["dDir3"], "Wind dir 3h\n(deg, - = backing)", BACKING_3H, f"backing <= {BACKING_3H}", below=True)
panel(ax2[3], f["dRH3"],  "RH 3h change\n(%)",    RH_RISE_3H, f"rise >= {RH_RISE_3H}")
panel(ax2[4], f["rain3"], "Rain 3h\n(mm)",        RAIN_3H,    f"rain >= {RAIN_3H}")
ax2[5].fill_between(f.index, f["pre_signs"], step="mid", color="orange", alpha=.6)
ax2[5].axhline(MIN_EXTRA, color="red", ls="--", lw=1, label=f"need >= {MIN_EXTRA} signs")
ax2[5].axhline(0, color="k", lw=.5)
ax2[5].set_ylabel("Pre-frontal\nsigns (0-4)"); ax2[5].legend(loc="upper left", fontsize=8); ax2[5].grid(alpha=.3)
panel(ax2[6], f["dP1"],   "P 1h change\n(hPa)",   P_FLAT_1H,  f"fall stopped >= {P_FLAT_1H}")
code = f["state"].map({"NO SIGNAL": 0, "BEFORE": 1, "PASSED": 2})
ax2[7].step(f.index, code, where="post", color="k")
ax2[7].fill_between(f.index, code, step="post", color="orange", alpha=.3, where=code == 1)
ax2[7].fill_between(f.index, code, step="post", color="green",  alpha=.3, where=code == 2)
ax2[7].set_yticks([0, 1, 2]); ax2[7].set_yticklabels(["NO SIGNAL", "BEFORE", "PASSED"])
ax2[7].set_ylabel("State"); ax2[7].grid(alpha=.3)
ax2[0].set_title("Indicators vs thresholds (red shading = threshold met)")
plt.tight_layout(); plt.savefig("before_warm_front_indicators.png", dpi=150)
plt.show()
f.to_csv("before_warm_front_states.csv")