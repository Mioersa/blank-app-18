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

# Five tabs (added chart tabs)
tab_fut, tab_opt, tab_comb, tab_fut_chart, tab_opt_chart = st.tabs([
    "📈 Futures", "📊 Options", "🪄 Composite Signals",
    "📉 Futures Chart", "📈 Options Chart"
])

# ============================================================
# (1) FUTURES TAB — existing logic retained
# ============================================================
with tab_fut:
    st.subheader("📈 Futures Multi‑Metric Analysis")
    fut_files = st.file_uploader("Upload 5‑min Futures CSVs", type="csv", accept_multiple_files=True)
    if not fut_files:
        st.info("⬅️ Upload Futures CSVs to continue.")
        combined_fut = pd.DataFrame()
    else:
        dfs=[]
        for uploaded in fut_files:
            fn=uploaded.name
            m=re.search(r"_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})\.csv$", fn)
            if m:
                dd, mm, yyyy, HH, MM, SS = m.groups()
                base_time = datetime(int(yyyy), int(mm), int(dd), int(HH), int(MM), int(SS))
            else:
                base_time = datetime.now()
            df=pd.read_csv(uploaded)
            df["timestamp"]=base_time + pd.to_timedelta(np.arange(len(df))*5,unit="min")
            dfs.append(df)
        fut_df=pd.concat(dfs,ignore_index=True)
        if "expiryDate" not in fut_df.columns:
            st.error("❌ Column 'expiryDate' not found in Futures data.")
            st.stop()
        expiry_opts=sorted(fut_df["expiryDate"].unique())
        expiry=st.selectbox("Select expiry",expiry_opts)
        fut_df=fut_df[fut_df["expiryDate"]==expiry].copy()

        def compute_indicators(df, metric):
            if metric not in df.columns:
                return pd.DataFrame()
            df=df.sort_values("timestamp").reset_index(drop=True)
            out=pd.DataFrame()
            out["time"]=df["timestamp"]
            out[f"Δ {metric}"]=df[metric].diff()
            out["Δ Price"]=df["lastPrice"].diff()
            out["SMAΔ"]=out[f"Δ {metric}"].rolling(rolling_n,min_periods=1).mean()
            out["RollCorr"]=out["Δ Price"].rolling(rolling_n).corr(out[f"Δ {metric}"])
            out["Signal"]=np.where(
                out["RollCorr"]>0,
                np.where(out["SMAΔ"]>0,"🟢 Bullish","⚪ Weak Up"),
                np.where(out["SMAΔ"]<0,"🔴 Bearish","⚪ Neutral"),
            )
            return out

        vol_df=compute_indicators(fut_df,"volume")
        oi_df=compute_indicators(fut_df,"openInterest")
        turn_df=compute_indicators(fut_df,"totalTurnover")

        combined_fut=pd.DataFrame({"time":vol_df["time"]})
        combined_fut["Vol_Signal"]=vol_df["Signal"]
        combined_fut["OI_Signal"]=oi_df["Signal"]
        combined_fut["Turn_Signal"]=turn_df["Signal"]

        def overall(r):
            vals=[r.get("Vol_Signal"),r.get("OI_Signal"),r.get("Turn_Signal")]
            if all(v and "🟢" in v for v in vals): return "🟢 Bullish"
            if all(v and "🔴" in v for v in vals): return "🔴 Bearish"
            return "⚪ Neutral"
        combined_fut["Overall_Fut_Signal"]=combined_fut.apply(overall,axis=1)

        st.dataframe(combined_fut.tail(20))

        # quick inline chart (original)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=vol_df["time"],y=vol_df["SMAΔ"],name="Δ Volume SMA",line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=oi_df["time"],y=oi_df["SMAΔ"],name="Δ OI SMA",line=dict(color="teal")))
        fig.add_trace(go.Scatter(x=turn_df["time"],y=turn_df["SMAΔ"],name="Δ Turnover SMA",line=dict(color="purple")))
        fig.update_layout(title="Rolling Δ Metrics (Volume / OI / Turnover)",height=400)
        st.plotly_chart(fig,use_container_width=True)

# ============================================================
# (2) OPTIONS TAB — existing logic retained
# ============================================================
with tab_opt:
    st.subheader("📊 Option‑Chain Rule‑Based Signals")
    opt_files=st.file_uploader("Upload 5‑min Option‑Chain CSVs",type="csv",accept_multiple_files=True)
    if not opt_files: 
        st.info("⬅️ Upload Option CSVs to continue.")
        opt_feat=pd.DataFrame()
    else:
        frames=[]
        for f in opt_files:
            base=f.name.replace(".csv","")
            try:
                ts=datetime.strptime(base.split("_")[-2]+"_"+base.split("_")[-1],"%d%m%Y_%H%M%S")
            except: ts=datetime.now()
            d=pd.read_csv(f); d["timestamp"]=ts; frames.append(d)
        df=pd.concat(frames,ignore_index=True)
        df=df[(df["CE_buyPrice1"]>0)&(df["CE_sellPrice1"]>0)]
        df["mid_CE"]=(df["CE_buyPrice1"]+df["CE_sellPrice1"])/2
        df["mid_PE"]=(df["PE_buyPrice1"]+df["PE_sellPrice1"])/2
        df["spread_pct"]=abs(df["CE_sellPrice1"]-df["CE_buyPrice1"])/df["mid_CE"]
        df=df[df["spread_pct"]<spread_cutoff]
        agg=df.groupby("timestamp").agg({
            "CE_lastPrice":"mean","PE_lastPrice":"mean",
            "CE_changeinOpenInterest":"sum","PE_changeinOpenInterest":"sum",
            "CE_impliedVolatility":"mean","PE_impliedVolatility":"mean"
        }).sort_index()
        agg["ΔPrice_CE"]=agg["CE_lastPrice"].diff()
        agg["ΔOI_CE"]=agg["CE_changeinOpenInterest"].diff()
        agg["ΔIV"]=agg["CE_impliedVolatility"].diff()
        agg["PCR_OI"]=agg["PE_changeinOpenInterest"]/(agg["CE_changeinOpenInterest"].replace(0,np.nan))
        agg["Volume_spike"]=(agg["CE_changeinOpenInterest"].abs()+agg["PE_changeinOpenInterest"].abs())/agg["CE_changeinOpenInterest"].abs().rolling(rolling_n).mean()
        def detect(row):
            if row["ΔPrice_CE"]*row["ΔOI_CE"]>0 and row["Volume_spike"]>1: reg="trend"
            elif abs(row["ΔPrice_CE"])<0.05: reg="range"
            elif row["ΔIV"]>0.2: reg="breakout"
            else: reg="quiet"
            return reg
        agg["regime"]=agg.apply(detect,axis=1)
        opt_feat=agg.reset_index()
        st.dataframe(opt_feat.tail(10))

# ============================================================
# (3) COMPOSITE TAB — same logic retained
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
    fut_m, opt_m = fut_m.sort_values("time_key"), opt_m.sort_values("time_key")

    merged=pd.merge_asof(fut_m,opt_m,on="time_key",direction="nearest",allow_exact_matches=True)

    def comp_sig(r):
        f=r["Overall_Fut_Signal"]
        m=abs(r.get("ΔPrice_CE",0))*50
        if "🟢" in f and m>20: return "🟢 Confirmed Uptrend"
        if "🔴" in f and m>20: return "🔴 Confirmed Downtrend"
        return "⚪ Mixed/Neutral"
    merged["Momentum_Score"]=abs(merged.get("ΔPrice_CE",0))*50
    merged["Composite_Signal"]=merged.apply(comp_sig,axis=1)
    st.dataframe(merged[["time","Overall_Fut_Signal","Momentum_Score","Composite_Signal"]])

# ============================================================
# (4) FUTURES CHART TAB — taken from your second script
# ============================================================
with tab_fut_chart:
    st.subheader("📉 Futures Chart – Δ Volume vs Last Price")
    files = st.file_uploader("Upload Futures CSV(s)",type="csv",accept_multiple_files=True,key="fut_chart_up")
    if not files:
        st.info("Upload CSV to plot.")
        st.stop()
    dfs, labels, times = [], [], []
    for f in files:
        fn=f.name
        m=re.search(r"_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})",fn)
        base=datetime.now(); lbl=fn
        if m:
            d,mo,y,h,mi,s=m.groups()
            base=datetime(int(y),int(mo),int(d),int(h),int(mi),int(s))
            lbl=f"{h}:{mi}"
        labels.append(lbl); times.append(base)
        df=pd.read_csv(f); df["timestamp"]=base+pd.to_timedelta(np.arange(len(df))*5,unit="min"); dfs.append(df)
    df_sum=pd.DataFrame()
    for lbl, df in zip(labels, dfs):
        df_sum=pd.concat([df_sum,pd.DataFrame({
            "time":[lbl],
            "volume":[df["volume"].iloc[0]],
            "last_price":[df["lastPrice"].iloc[-1]]
        })])
    df_sum["Δ Volume"]=df_sum["volume"].diff(); df_sum["Δ Price"]=df_sum["last_price"].diff()
    axis_type=st.radio("Y‑axis scale",["linear","log"],horizontal=True,key="futchart_scale")
    fig=go.Figure()
    fig.add_trace(go.Bar(x=df_sum["time"],y=df_sum["Δ Volume"],name="Δ Volume",marker_color="orange",opacity=0.6,yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_sum["time"],y=df_sum["last_price"],mode="lines+markers",line=dict(color="blue"),name="Last Price",yaxis="y1"))
    fig.update_layout(height=600,margin=dict(l=60,r=40,t=60,b=60),
        xaxis=dict(title="Capture Time (HH:MM)",rangeslider=dict(visible=True)),
        yaxis=dict(domain=[0.45,1.0],title="Last Price"),
        yaxis2=dict(domain=[0,0.35],title="Δ Volume",type=axis_type),
        title="Chart – Δ Volume vs Last Price",hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)

# ============================================================
# (5) OPTIONS CHART TAB — taken from your second script
# ============================================================
with tab_opt_chart:
    st.subheader("📈 Options Chart Viewer – Δ Metrics per Strike")
    files=st.file_uploader("Upload Option CSVs (_DDMMYYYY_HHMMSS.csv)",
                           type=["csv"],accept_multiple_files=True,key="opt_chart_up")
    if not files:
        st.info("Upload option CSVs to plot."); st.stop()

    def parse(name):
        m=re.search(r"_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})",name)
        if not m: return None
        d,mo,y,h,mi,s=m.groups()
        return f"{d}-{mo}-{y} {h}:{mi}:{s}"

    fr=[]
    for f in files:
        d=pd.read_csv(f); d["timestamp"]=parse(f.name); fr.append(d)
    df=pd.concat(fr)
    df["timestamp"]=pd.to_datetime(df["timestamp"],format="%d-%m-%Y %H:%M:%S",errors="coerce")
    df=df.sort_values("timestamp")

    strikes=sorted(pd.to_numeric(df.get("CE_strikePrice",df.get("PE_strikePrice",pd.Series())),
                                 errors="coerce").dropna().unique())
    if not strikes: st.warning("No strike data."); st.stop()
    strike=st.selectbox("Select Strike Price",strikes)
    opt=st.radio("Option Type",["CE","PE"],horizontal=True)
    metric=st.selectbox("Metric",["lastPrice","openInterest","totalTradedVolume","impliedVolatility"])
    col=f"{opt}_{metric}"
    if col not in df:
        st.warning(f"'{col}' missing in data.")
    else:
        sub=df[df[f"{opt}_strikePrice"]==strike]
        if sub.empty: st.warning("No data for that strike.")
        else:
            fig=px.line(sub,x="timestamp",y=col,markers=True,title=f"{opt} {metric} for Strike {strike}")
            st.plotly_chart(fig,use_container_width=True)

st.caption("✅ Full Engine — Futures + Options logic retained with added Charts for visualization.")
