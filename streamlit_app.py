import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.graph_objects as go
import altair as alt

st.set_page_config(page_title="🧩 Futures + Options Intraday Signal Engine", layout="wide")
st.title("🧩 Futures + Options Intraday Signal Engine")

# ============================================================
#  SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Global Parameters")
rolling_n = st.sidebar.number_input("Rolling window (mins)", 3, 60, 5)
spread_cutoff = st.sidebar.slider("Max Option bid‑ask spread %", 0.0, 1.0, 0.2)
num_strikes = st.sidebar.number_input("Top Option strikes by OI", 1, 30, 6)

tab_fut, tab_opt, tab_comb = st.tabs(["📈 Futures", "📊 Options", "🪄 Composite Signals"])

# ============================================================
#  FUTURES TAB
# ============================================================
with tab_fut:
    st.subheader("📈 Futures Multi‑Metric Analysis")

    fut_files = st.file_uploader(
        "Upload 5‑min Futures CSVs", type="csv", accept_multiple_files=True
    )
    if not fut_files:
        st.info("⬅️ Upload Futures CSVs to continue.")
        fut_df = vol_df = oi_df = turn_df = combined_fut = pd.DataFrame()
    else:
        dfs, labels, times = [], [], []
        for upl in fut_files:
            fn = upl.name
            m = re.search(r"_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})\.csv$", fn)
            if m:
                dd, mm, yyyy, HH, MM, SS = m.groups()
                base = datetime(int(yyyy), int(mm), int(dd), int(HH), int(MM), int(SS))
                label = f"{HH}:{MM}"
            else:
                base, label = datetime.now(), upl.name
            times.append(base)
            labels.append(label)
            df = pd.read_csv(upl)
            df["timestamp"] = base + pd.to_timedelta(np.arange(len(df)) * 5, unit="min")
            dfs.append(df)

        fut_df = pd.concat(dfs)
        expiry_opts = sorted(fut_df["expiryDate"].unique())
        expiry = st.selectbox("Select expiry", expiry_opts)
        fut_df = fut_df[fut_df["expiryDate"] == expiry]

        # ---- metric computation
        def compute_ind(df, metric):
            if metric not in df.columns:
                return pd.DataFrame()
            recs = []
            for lbl, base in zip(labels, times):
                sub = df.copy()
                val = sub[metric].iloc[0]
                price = sub["lastPrice"].iloc[-1]
                recs.append(
                    {"time": lbl, metric: val, "Δ Price": sub["lastPrice"].diff().sum()}
                )
            out = pd.DataFrame(recs)
            out[f"Δ {metric}"] = out[metric].diff()
            out["SMAΔ"] = out[f"Δ {metric}"].rolling(3).mean()
            out["Signal"] = np.where(
                out["SMAΔ"] > 0,
                "🟢 Bullish",
                np.where(out["SMAΔ"] < 0, "🔴 Bearish", "⚪ Neutral"),
            )
            return out

        vol_df = compute_ind(fut_df, "volume")
        oi_df = compute_ind(fut_df, "openInterest")
        turn_df = compute_ind(fut_df, "totalTurnover")

        # ---- combined
        combined_fut = pd.DataFrame({"time": vol_df["time"]})
        combined_fut["Vol_Signal"] = vol_df["Signal"]
        combined_fut["OI_Signal"] = oi_df["Signal"]
        combined_fut["Turn_Signal"] = turn_df["Signal"]

        def overall(r):
            vals = [r["Vol_Signal"], r["OI_Signal"], r["Turn_Signal"]]
            if all(v == "🟢 Bullish" for v in vals):
                return "🟢 Bullish"
            if all(v == "🔴 Bearish" for v in vals):
                return "🔴 Bearish"
            return "⚪ Neutral"

        combined_fut["Overall_Fut_Signal"] = combined_fut.apply(overall, axis=1)
        st.dataframe(combined_fut)

# ============================================================
#  OPTIONS TAB
# ============================================================
with tab_opt:
    st.subheader("📊 Option‑Chain Rule‑Based Signals")

    opt_files = st.file_uploader(
        "Upload 5‑min Option‑Chain CSVs", type="csv", accept_multiple_files=True
    )
    if not opt_files:
        st.info("⬅️ Upload Option CSVs to continue.")
        opt_feat = pd.DataFrame()
    else:
        frames = []
        for f in opt_files:
            base = f.name.replace(".csv", "")
            try:
                ts = datetime.strptime(
                    base.split("_")[-2] + "_" + base.split("_")[-1], "%d%m%Y_%H%M%S"
                )
            except Exception:
                ts = datetime.now()
            d = pd.read_csv(f)
            d["timestamp"] = ts
            frames.append(d)

        raw = pd.concat(frames)
        df = raw.copy()
        df = df[(df["CE_buyPrice1"] > 0) & (df["CE_sellPrice1"] > 0)]
        df["mid_CE"] = (df["CE_buyPrice1"] + df["CE_sellPrice1"]) / 2
        df["mid_PE"] = (df["PE_buyPrice1"] + df["PE_sellPrice1"]) / 2
        df["spread_pct"] = abs(df["CE_sellPrice1"] - df["CE_buyPrice1"]) / df["mid_CE"]
        df = df[df["spread_pct"] < spread_cutoff]

        if "CE_expiryDate" in df:
            df["CE_expiryDate"] = pd.to_datetime(df["CE_expiryDate"], errors="coerce")
            df["days_to_expiry"] = (
                df["CE_expiryDate"] - df["timestamp"]
            ).dt.days.fillna(1)
        else:
            df["days_to_expiry"] = 1

        df["θ_adj_CE"] = df["CE_lastPrice"] / np.sqrt(df["days_to_expiry"].clip(lower=1))
        df["θ_adj_PE"] = df["PE_lastPrice"] / np.sqrt(df["days_to_expiry"].clip(lower=1))

        agg = (
            df.groupby("timestamp")
            .agg(
                {
                    "CE_lastPrice": "mean",
                    "PE_lastPrice": "mean",
                    "CE_changeinOpenInterest": "sum",
                    "PE_changeinOpenInterest": "sum",
                    "CE_totalTradedVolume": "sum",
                    "PE_totalTradedVolume": "sum",
                    "CE_impliedVolatility": "mean",
                    "PE_impliedVolatility": "mean",
                }
            )
            .sort_index()
        )

        agg["ΔPrice_CE"] = agg["CE_lastPrice"].diff()
        agg["ΔOI_CE"] = agg["CE_changeinOpenInterest"].diff()
        agg["ΔIV"] = agg["CE_impliedVolatility"].diff()
        agg["PCR_OI"] = agg["PE_changeinOpenInterest"] / (
            agg["CE_changeinOpenInterest"].replace(0, np.nan)
        )
        agg["Volume_spike"] = (
            (agg["CE_totalTradedVolume"] + agg["PE_totalTradedVolume"])
            / (agg["CE_totalTradedVolume"] + agg["PE_totalTradedVolume"])
            .rolling(rolling_n)
            .mean()
        )

        def detect(row):
            if row["ΔPrice_CE"] * row["ΔOI_CE"] > 0 and row["Volume_spike"] > 1:
                reg = "trend"
            elif abs(row["ΔPrice_CE"]) < 0.05:
                reg = "range"
            elif row["ΔIV"] > 0.2 and row["Volume_spike"] > 1.2:
                reg = "breakout"
            else:
                reg = "quiet"
            bias = (
                "bullish"
                if row["PCR_OI"] < 0.8
                else "bearish"
                if row["PCR_OI"] > 1.2
                else "neutral"
            )
            return reg, bias

        agg[["regime", "bias"]] = agg.apply(detect, axis=1, result_type="expand")

        def make_sig(r):
            if r["regime"] == "trend" and r["bias"] == "bullish":
                return "BUY_CALL"
            if r["regime"] == "trend" and r["bias"] == "bearish":
                return "BUY_PUT"
            if r["regime"] == "breakout":
                return "MOMENTUM_TRADE"
            if r["regime"] == "range":
                return "SELL_STRANGLE"
            return "HOLD"

        agg["Opt_Signal"] = agg.apply(make_sig, axis=1)
        opt_feat = agg.reset_index()
        st.dataframe(opt_feat.tail(10))

# ============================================================
#  COMPOSITE TAB
# ============================================================
with tab_comb:
    st.subheader("🪄 Composite Signals — Futures + Options Alignment")

    if "combined_fut" not in locals() or combined_fut.empty or opt_feat.empty:
        st.warning("Upload both Futures and Options files first.")
        st.stop()

    fut_m = combined_fut.copy()
    opt_m = opt_feat.copy()

    # --- FIX: ensure datetime keys for merge_asof
    def to_datetime_key(t):
        if isinstance(t, str):
            return pd.to_datetime(t, format="%H:%M", errors="coerce")
        return pd.to_datetime(t, errors="coerce")

    fut_m["time_key"] = fut_m["time"].apply(to_datetime_key)
    opt_m["time_key"] = pd.to_datetime(opt_m["timestamp"], errors="coerce")

    fut_m = fut_m.dropna(subset=["time_key"]).sort_values("time_key")
    opt_m = opt_m.dropna(subset=["time_key"]).sort_values("time_key")

    merged = pd.merge_asof(
        fut_m, opt_m, left_on="time_key", right_on="time_key", direction="nearest"
    )

    # ---- Composite logic
    def comp_sig(r):
        f = r["Overall_Fut_Signal"]
        b = r["bias"]
        if f.startswith("🟢") and b == "bullish":
            return "🟢 Confirmed Uptrend"
        if f.startswith("🔴") and b == "bearish":
            return "🔴 Confirmed Downtrend"
        if f.startswith("🟢") and b == "bearish":
            return "🟠 Futures↑ / Opts Bearish (Divergent)"
        if f.startswith("🔴") and b == "bullish":
            return "🟣 Futures↓ / Opts Bullish (Reversal Watch)"
        return "⚪ Neutral/Unclear"

    def momentum_score(r):
        sc = (
            abs(r.get("ΔPrice_CE", 0)) * 50
            + abs(r.get("ΔOI_CE", 0)) * 0.01
            + abs(r.get("ΔIV", 0)) * 30
            + (r.get("Volume_spike", 0) - 1) * 10
        )
        return round(np.clip(sc, 0, 100), 1)

    merged["Momentum_Score"] = merged.apply(momentum_score, axis=1)
    merged["Composite_Signal"] = merged.apply(comp_sig, axis=1)

    cols_to_show = [
        "time",
        "Overall_Fut_Signal",
        "bias",
        "Opt_Signal",
        "Momentum_Score",
        "Composite_Signal",
    ]
    st.dataframe(merged[cols_to_show])

    # chart
    st.subheader("Composite Momentum Chart")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=merged["time_key"],
            y=merged["Momentum_Score"],
            name="Momentum Score",
            marker_color="purple",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "⬇️ Download Composite Signals",
        merged[cols_to_show].to_csv(index=False).encode("utf‑8"),
        "composite_signals.csv",
        "text/csv",
    )

st.caption("Fusion engine combines Futures and Options bias + momentum for clear directional insights.")


