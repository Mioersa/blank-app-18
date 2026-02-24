import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import altair as alt

st.set_page_config(page_title="🧩 Futures + Options Intraday Signal Engine", layout="wide")
st.title("🧩 Futures + Options Intraday Signal Engine")

# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Global Parameters")
rolling_n = st.sidebar.number_input("Rolling window (mins)", 3, 60, 5)
spread_cutoff = st.sidebar.slider("Max Option bid‑ask spread %", 0.0, 1.0, 0.2)
num_strikes = st.sidebar.number_input("Top Option strikes by OI", 1, 30, 6)

tab_fut, tab_opt, tab_comb, tab_fut_chart, tab_opt_chart = st.tabs([
    "📈 Futures", "📊 Options", "🪄 Composite Signals",
    "📉 Futures Chart", "📈 Options Chart"
])

# ============================================================
# FUTURES TAB
# ============================================================
with tab_fut:
    st.subheader("📈 Futures Multi‑Metric Analysis")
    fut_files = st.file_uploader("Upload 5‑min Futures CSVs", type="csv", accept_multiple_files=True)

    if not fut_files:
        st.info("⬅️ Upload Futures CSVs to continue.")
        combined_fut = pd.DataFrame()
    else:
        dfs = []
        for uploaded in fut_files:
            fn = uploaded.name
            m = re.search(r"_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})\.csv$", fn)
            if m:
                dd, mm, yyyy, HH, MM, SS = m.groups()
                base_time = datetime(int(yyyy), int(mm), int(dd), int(HH), int(MM), int(SS))
            else:
                base_time = datetime.now()
            df = pd.read_csv(uploaded)
            df["timestamp"] = base_time + pd.to_timedelta(np.arange(len(df)) * 5, unit="min")
            dfs.append(df)

        fut_df = pd.concat(dfs, ignore_index=True)
        if "expiryDate" not in fut_df.columns:
            st.error("❌ Column 'expiryDate' not found in Futures data.")
            st.stop()

        expiry_opts = sorted(fut_df["expiryDate"].unique())
        expiry = st.selectbox("Select expiry", expiry_opts)
        fut_df = fut_df[fut_df["expiryDate"] == expiry].copy()

        def compute_indicators(df, metric):
            if metric not in df.columns:
                return pd.DataFrame()
            df = df.sort_values("timestamp").reset_index(drop=True)
            out = pd.DataFrame()
            out["time"] = df["timestamp"]
            out[f"Δ {metric}"] = df[metric].diff()
            out["Δ Price"] = df["lastPrice"].diff()
            out["SMAΔ"] = out[f"Δ {metric}"].rolling(5, min_periods=1).mean()
            out["RollCorr"] = out["Δ Price"].rolling(5).corr(out[f"Δ {metric}"])
            out["Signal"] = np.where(
                out["RollCorr"] > 0,
                np.where(out["SMAΔ"] > 0, "🟢 Bullish", "⚪ Weak Up"),
                np.where(out["SMAΔ"] < 0, "🔴 Bearish", "⚪ Neutral"),
            )
            return out

        vol_df = compute_indicators(fut_df, "volume")
        oi_df = compute_indicators(fut_df, "openInterest")
        turn_df = compute_indicators(fut_df, "totalTurnover")

        combined_fut = pd.DataFrame({"time": vol_df["time"]})
        combined_fut["Vol_Signal"] = vol_df["Signal"]
        combined_fut["OI_Signal"] = oi_df["Signal"]
        combined_fut["Turn_Signal"] = turn_df["Signal"]

        def overall(r):
            vals = [r.get("Vol_Signal"), r.get("OI_Signal"), r.get("Turn_Signal")]
            if all(v and "🟢" in v for v in vals): return "🟢 Bullish"
            if all(v and "🔴" in v for v in vals): return "🔴 Bearish"
            return "⚪ Neutral"
        combined_fut["Overall_Fut_Signal"] = combined_fut.apply(overall, axis=1)

        st.dataframe(combined_fut.tail(20))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vol_df["time"], y=vol_df["SMAΔ"], name="Δ Volume SMA", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=oi_df["time"], y=oi_df["SMAΔ"], name="Δ OI SMA", line=dict(color="teal")))
        fig.add_trace(go.Scatter(x=turn_df["time"], y=turn_df["SMAΔ"], name="Δ Turnover SMA", line=dict(color="purple")))
        fig.update_layout(title="Rolling Δ Metrics (Volume / OI / Turnover)", height=400,
                          margin=dict(l=50, r=30, t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# OPTIONS TAB
# ============================================================
with tab_opt:
    st.subheader("📊 Option‑Chain Rule‑Based Signals")
    opt_files = st.file_uploader("Upload 5‑min Option‑Chain CSVs", type="csv", accept_multiple_files=True)
    if not opt_files:
        st.info("⬅️ Upload Option CSVs to continue.")
        opt_feat = pd.DataFrame()
    else:
        frames = []
        for f in opt_files:
            base = f.name.replace(".csv", "")
            try:
                ts = datetime.strptime(base.split("_")[-2] + "_" + base.split("_")[-1], "%d%m%Y_%H%M%S")
            except Exception:
                ts = datetime.now()
            d = pd.read_csv(f); d["timestamp"] = ts; frames.append(d)
        df = pd.concat(frames, ignore_index=True)

        df = df[(df["CE_buyPrice1"] > 0) & (df["CE_sellPrice1"] > 0)]
        df["mid_CE"] = (df["CE_buyPrice1"] + df["CE_sellPrice1"]) / 2
        df["mid_PE"] = (df["PE_buyPrice1"] + df["PE_sellPrice1"]) / 2
        df["spread_pct"] = abs(df["CE_sellPrice1"] - df["CE_buyPrice1"]) / df["mid_CE"]
        df = df[df["spread_pct"] < spread_cutoff]
        if "CE_expiryDate" in df:
            df["CE_expiryDate"] = pd.to_datetime(df["CE_expiryDate"], errors="coerce")
            df["days_to_expiry"] = (df["CE_expiryDate"] - df["timestamp"]).dt.days.fillna(1)
        else:
            df["days_to_expiry"] = 1
        df["θ_adj_CE"] = df["CE_lastPrice"] / np.sqrt(df["days_to_expiry"].clip(lower=1))
        df["θ_adj_PE"] = df["PE_lastPrice"] / np.sqrt(df["days_to_expiry"].clip(lower=1))

        agg = df.groupby("timestamp").agg({
            "CE_lastPrice":"mean","PE_lastPrice":"mean",
            "CE_changeinOpenInterest":"sum","PE_changeinOpenInterest":"sum",
            "CE_totalTradedVolume":"sum","PE_totalTradedVolume":"sum",
            "CE_impliedVolatility":"mean","PE_impliedVolatility":"mean"
        }).sort_index()

        agg["ΔPrice_CE"]=agg["CE_lastPrice"].diff()
        agg["ΔOI_CE"]=agg["CE_changeinOpenInterest"].diff()
        agg["ΔIV"]=agg["CE_impliedVolatility"].diff()
        agg["PCR_OI"]=agg["PE_changeinOpenInterest"]/(agg["CE_changeinOpenInterest"].replace(0,np.nan))
        agg["Volume_spike"]=(agg["CE_totalTradedVolume"]+agg["PE_totalTradedVolume"]) / \
            (agg["CE_totalTradedVolume"]+agg["PE_totalTradedVolume"]).rolling(rolling_n).mean()

        def detect(row):
            if row["ΔPrice_CE"]*row["ΔOI_CE"]>0 and row["Volume_spike"]>1: reg="trend"
            elif abs(row["ΔPrice_CE"])<0.05: reg="range"
            elif row["ΔIV"]>0.2 and row["Volume_spike"]>1.2: reg="breakout"
            else: reg="quiet"
            bias="bullish" if row["PCR_OI"]<0.8 else ("bearish" if row["PCR_OI"]>1.2 else "neutral")
            return reg,bias
        agg[["regime","bias"]]=agg.apply(detect,axis=1,result_type="expand")

        def make_sig(r):
            if r["regime"]=="trend" and r["bias"]=="bullish": return "BUY_CALL"
            if r["regime"]=="trend" and r["bias"]=="bearish": return "BUY_PUT"
            if r["regime"]=="breakout": return "MOMENTUM_TRADE"
            if r["regime"]=="range": return "SELL_STRANGLE"
            return "HOLD"
        agg["Opt_Signal"]=agg.apply(make_sig,axis=1)
        opt_feat=agg.reset_index()
        st.dataframe(opt_feat.tail(10))

# ============================================================
# COMPOSITE TAB
# ============================================================
with tab_comb:
    st.subheader("🪄 Composite Signals — Futures + Options Alignment")

    if "combined_fut" not in locals() or combined_fut.empty or opt_feat.empty:
        st.warning("Upload both Futures and Options files first.")
        st.stop()

    fut_m=combined_fut.copy()
    opt_m=opt_feat.copy()

    fut_m["time_key"]=pd.to_datetime(fut_m["time"],errors="coerce").dt.tz_localize(None)
    opt_m["time_key"]=pd.to_datetime(opt_m["timestamp"],errors="coerce").dt.tz_localize(None)
    fut_m=fut_m.dropna(subset=["time_key"]).sort_values("time_key")
    opt_m=opt_m.dropna(subset=["time_key"]).sort_values("time_key")

    merged=pd.merge_asof(fut_m,opt_m,on="time_key",direction="nearest",allow_exact_matches=True)

    def comp_sig(r):
        f=r["Overall_Fut_Signal"]; b=r["bias"]
        if isinstance(f,str) and f.startswith("🟢") and b=="bullish": return "🟢 Confirmed Uptrend"
        if isinstance(f,str) and f.startswith("🔴") and b=="bearish": return "🔴 Confirmed Downtrend"
        if isinstance(f,str) and f.startswith("🟢") and b=="bearish": return "🟠 Futures↑ / Opts Bearish (Divergent)"
        if isinstance(f,str) and f.startswith("🔴") and b=="bullish": return "🟣 Futures↓ / Opts Bullish (Reversal Watch)"
        return "⚪ Neutral/Unclear"

    def momentum_score(r):
        sc=(abs(r.get("ΔPrice_CE",0))*50 +
            abs(r.get("ΔOI_CE",0))*0.01 +
            abs(r.get("ΔIV",0))*30 +
            (r.get("Volume_spike",0)-1)*10)
        return round(np.clip(sc,0,100),1)

    merged["Momentum_Score"]=merged.apply(momentum_score,axis=1)
    merged["Composite_Signal"]=merged.apply(comp_sig,axis=1)

    cols_to_show=["time","Overall_Fut_Signal","bias","Opt_Signal","Momentum_Score","Composite_Signal"]
    st.dataframe(merged[cols_to_show])

    st.subheader("Composite Momentum Chart")
    fig=go.Figure()
    fig.add_trace(go.Bar(x=merged["time_key"],y=merged["Momentum_Score"],
                         name="Momentum Score",marker_color="purple"))
    st.plotly_chart(fig,use_container_width=True)

    st.download_button(
        "⬇️ Download Composite Signals",
        merged[cols_to_show].to_csv(index=False).encode("utf‑8"),
        "composite_signals.csv","text/csv")
    st.caption("Fusion engine combines Futures and Options bias + momentum for clear directional insights.")

# ============================================================
# FUTURES CHART TAB
# ============================================================
with tab_fut_chart:
    st.subheader("📉 Futures Chart – Δ Volume vs Last Price")
    files=st.file_uploader("Upload Futures CSV file(s)",type="csv",accept_multiple_files=True,key="fut_chart_up")
    if not files:
        st.info("Upload CSV to plot."); st.stop()
    dfs,labels,times=[],[],[]
    for f in files:
        fn=f.name
        m=re.search(r"_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})",fn)
        base=datetime.now(); lbl=fn
        if m:
            d,mo,y,h,mi,s=m.groups()
            base=datetime(int(y),int(mo),int(d),int(h),int(mi),int(s))
            lbl=f"{h}:{mi}"
        labels.append(lbl); times.append(base)
        df=pd.read_csv(f)
        df["timestamp"]=base+pd.to_timedelta(np.arange(len(df))*5,unit="min")
        dfs.append(df)
    first=dfs[0]
    if "expiryDate" not in first:
        st.error("expiryDate missing."); st.stop()
    exp_opts=sorted(first["expiryDate"].unique())
    exp=st.selectbox("Select Expiry",exp_opts)
    filt=[]
    for i,dff in enumerate(dfs):
        sub=dff[dff["expiryDate"]==exp].copy()
        if sub.empty: continue
        sub["label"]=labels[i]; sub["cap_time"]=times[i]; filt.append(sub)
    if not filt: st.stop()
    final=pd.concat(filt).sort_values(["contract","timestamp"]).reset_index(drop=True)
    rows=[]
    for lbl in labels:
        sub=final[final["label"]==lbl]
        if sub.empty: continue
        vol=sub["volume"].iloc[0]; price=sub["lastPrice"].iloc[-1]
        rows.append({"time":lbl,"volume":vol,"last_price":price})
    df_sum=pd.DataFrame(rows)
    df_sum["Δ Volume"]=df_sum["volume"].diff()
    df_sum["Δ Price"]=df_sum["last_price"].diff()
    axis_type=st.radio("Y‑axis scale",["linear","log"],horizontal=True,key="futchart_scale")
    config={"scrollZoom":True,"displaylogo":False}
    fig=go.Figure()
    fig.add_trace(go.Bar(x=df_sum["time"],y=df_sum["Δ Volume"],
                         name="Δ Volume",marker_color="orange",opacity=0.6,yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_sum["time"],y=df_sum["last_price"],
                             mode="lines+markers",line=dict(color="blue"),
                             name="Last Price",yaxis="y1"))
    fig.update_layout(height=600,margin=dict(l=60,r=40,t=60,b=60),
                      xaxis=dict(title="Capture Time (HH:MM)",rangeslider=dict(visible=True)),
                      yaxis=dict(domain=[0.45,1.0],title="Last Price"),
                      yaxis2=dict(domain=[0,0.35],title="Δ Volume",type=axis_type),
                      title="Chart 2 – Δ Volume vs Last Price",
                      legend=dict(orientation="h"),hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True,config=config)

# ============================================================
# OPTIONS CHART TAB
# ============================================================
with tab_opt_chart:
    st.subheader("📈 Options Chart Viewer (Δ metrics per strike)")
    files=st.file_uploader("Upload Option CSVs (_DDMMYYYY_HHMMSS.csv)",
                           type=["csv"],accept_multiple_files=True,key="opt_chart_up")
    if not files:
        st.info("Upload option CSVs to plot."); st.stop()

    def parse(name):
        m=re.search(r"_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})",name)
        if not m: return None
        d,mo,y,h,mi,s=m.groups(); return f"{d}-{mo}-{y} {h}:{mi}:{s}"

    fr=[]
    for f in files:
        d=pd.read_csv(f); d["timestamp"]=parse(f.name); fr.append(d)
    df=pd.concat(fr)
    df["timestamp"]=pd.to_datetime(df["timestamp"],format="%d-%m-%Y %H:%M:%S",errors="coerce")
    df=df.sort_values("timestamp")

    strikes=sorted(pd.to_numeric(df.get("CE_strikePrice",df.get("PE_strikePrice",pd.Series())),
                                 errors="coerce").dropna().unique())
    if not strikes: st.warning("No strike data."); st.stop()
    strike=st.selectbox("Select Strike",strikes)
    opt=st.radio("Option Type",["CE","PE"],horizontal=True)
    metric=st.selectbox("Metric",
                        ["lastPrice","openInterest","totalTradedVolume","impliedVolatility"])
    col=f"{opt}_{metric}"
    if col not in df:
        st.warning(f"'{col}' missing in data.")
    else:
        sub=df[df.get(f"{opt}_strikePrice")==strike]
        if sub.empty: st.warning("No data for that strike.")
        else:
            fig=px.line(sub,x="timestamp",y=col,markers=True,
                        title=f"{opt} {metric} for Strike {strike}")
            st.plotly_chart(fig,use_container_width=True)

st.caption("✅ All five modules integrated — Futures + Options analysis with Composite and dual chart tabs.")


