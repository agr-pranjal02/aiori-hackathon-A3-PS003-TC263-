"""SmartFlow FL - Federated Traffic Flow Prediction & Communication Cost Analyzer.
Run with:  streamlit run app.py   (federated clients are SIMULATED on one computer)"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.generate_data import generate_traffic_data, REQUIRED
from models.traffic_model import FEATURES, prepare, split_by_junction, xy, LinearModel, train_forest, SCALE
from models.federated import run_federated
from utils.metrics import mae, rmse
from utils.communication import fmt_bytes, params_bytes

st.set_page_config(page_title="SmartFlow FL", page_icon="🚦", layout="wide")
st.markdown("""<style>
div[data-testid="stMetric"]{background:linear-gradient(135deg,#131c2e,#1a2740);border:1px solid #243554;
border-left:4px solid #00d4aa;padding:14px;border-radius:12px}
h1{letter-spacing:-1px}.sim{background:#3a2f0b;color:#ffd166;padding:6px 12px;border-radius:8px;font-size:.85rem}
</style>""", unsafe_allow_html=True)
CHART = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
FINAL_MSG = ("SmartFlow FL demonstrates how distributed traffic locations can collaboratively improve "
             "traffic-flow prediction while keeping local datasets at their respective clients and "
             "explicitly measuring the communication required for federated training.")

# ---------------- Data loading (sample data or user CSV) ----------------
st.sidebar.title("🚦 SmartFlow FL")
page = st.sidebar.radio("Navigate", ["Overview", "Traffic Prediction", "Federated Learning",
                                     "Communication Cost", "Model Performance", "About the Project"])
up = st.sidebar.file_uploader("Optional: upload your traffic CSV", type="csv")
st.sidebar.caption("Required columns: " + ", ".join(REQUIRED))

@st.cache_data
def sample():
    return generate_traffic_data()

raw = sample()
if up is not None:
    try:
        u = pd.read_csv(up, parse_dates=["timestamp"])
        missing = [c for c in REQUIRED if c not in u.columns]
        if missing:
            st.sidebar.error(f"CSV is missing columns: {missing}. Using sample data.")
        elif u["junction_id"].nunique() < 2 or len(u) < 200:
            st.sidebar.error("CSV needs at least 2 junctions and 200 rows. Using sample data.")
        else:
            raw = u; st.sidebar.success("Using your uploaded CSV")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV ({e}). Using sample data.")

junctions = sorted(raw["junction_id"].unique())[:4]
raw = raw[raw["junction_id"].isin(junctions)]
data = prepare(raw)
train, test = split_by_junction(data)
ss = st.session_state
if ss.get("data_sig") != (len(raw), tuple(junctions)):   # reset results if data changed
    ss.clear(); ss["data_sig"] = (len(raw), tuple(junctions))

st.title("SmartFlow FL")
st.caption("Federated Traffic Flow Prediction & Communication Cost Analyzer")

# ---------------- Pages ----------------
if page == "Overview":
    cen, fed = ss.get("cen"), ss.get("fed")
    c = st.columns(3)
    c[0].metric("Junctions", len(junctions))
    c[1].metric("Federated rounds", len(fed["history"]) if fed else 0)
    c[2].metric("Current predicted traffic", f"{cen['next']:.0f} veh/h" if cen else "Run prediction")
    c = st.columns(3)
    c[0].metric("Model error (MAE)", f"{fed['history'][-1]['mae']:.2f}" if fed else
                (f"{cen['rf_mae']:.2f} (RF)" if cen else "—"))
    c[1].metric("Total communication", fmt_bytes(fed["history"][-1]["cumulative"]) if fed else "—")
    c[2].metric("Training status", "Federated trained ✅" if fed else ("Centralized only" if cen else "Not started"))
    st.markdown('<span class="sim">Federated clients are SIMULATED on one computer.</span>', unsafe_allow_html=True)
    d = raw.copy(); d["timestamp"] = pd.to_datetime(d["timestamp"])
    st.plotly_chart(px.line(d[d["timestamp"] < d["timestamp"].min() + pd.Timedelta(days=3)],
                            x="timestamp", y="vehicle_count", color="junction_id",
                            title="Traffic by junction (first 3 days)", **CHART), use_container_width=True)
    st.subheader("Hourly pattern per junction")
    st.plotly_chart(px.line(raw.groupby(["hour", "junction_id"])["vehicle_count"].mean().reset_index(),
                            x="hour", y="vehicle_count", color="junction_id", **CHART), use_container_width=True)
    with st.expander("View raw data"):
        st.dataframe(raw.head(200))

elif page == "Traffic Prediction":
    st.subheader("Centralized baseline")
    st.info("Baseline: all junction data is pooled in one place (this is what federated learning avoids).")
    if st.button("Run centralized prediction", type="primary"):
        with st.spinner("Training centralized models..."):
            Xtr, ytr = xy(train); Xte, yte = xy(test)
            rf = train_forest(Xtr, ytr); prf = rf.predict(Xte) * SCALE
            lin = LinearModel(); lin.fit(Xtr, ytr, epochs=40)   # same model family as FL
            pl = lin.predict(Xte) * SCALE
            last = test.groupby("junction_id").tail(1)
            ss["cen"] = dict(rf_mae=mae(yte * SCALE, prf), rf_rmse=rmse(yte * SCALE, prf),
                             lin_mae=mae(yte * SCALE, pl), lin_rmse=rmse(yte * SCALE, pl),
                             n=len(train), pred=prf, next=float(rf.predict(last[FEATURES].to_numpy()).mean() * SCALE),
                             raw_bytes=int(train[["vehicle_count", "average_speed", "occupancy", "hour", "day_of_week"]].to_numpy(np.float32).nbytes),
                             test=test.assign(pred=prf))
    cen = ss.get("cen")
    if cen:
        c = st.columns(4)
        c[0].metric("RF MAE", f"{cen['rf_mae']:.2f}"); c[1].metric("RF RMSE", f"{cen['rf_rmse']:.2f}")
        c[2].metric("Training rows", f"{cen['n']:,}"); c[3].metric("Linear MAE (FL-comparable)", f"{cen['lin_mae']:.2f}")
        j = st.selectbox("Junction", junctions)
        t = cen["test"][cen["test"]["junction_id"] == j]
        f = go.Figure()
        f.add_scatter(x=t["timestamp"], y=t["vehicle_count"], name="Actual")
        f.add_scatter(x=t["timestamp"], y=t["pred"], name="Predicted (RF)")
        f.update_layout(title=f"Actual vs predicted traffic - {j}", **CHART)
        st.plotly_chart(f, use_container_width=True)
    else:
        st.write("Click the button to train and see predictions.")

elif page == "Federated Learning":
    st.markdown('<span class="sim">SIMULATION: each client below is a separate local dataset in one Python process.</span>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    n_cl = c1.selectbox("Number of clients", [2, 3, 4], index=2)
    rounds = c2.selectbox("Federated rounds", [1, 5, 10, 20], index=2)
    names = junctions[:n_cl]
    st.write("Active clients:", ", ".join(f"Client {i+1} = {n}" for i, n in enumerate(names)))
    if st.button("Start Federated Training", type="primary"):
        clients = {n: xy(train[train["junction_id"] == n]) for n in names}
        tests = {n: xy(test[test["junction_id"] == n]) for n in names}
        bar, status, chart, tbl = st.progress(0.0), st.empty(), st.empty(), st.empty()
        def cb(h):
            bar.progress(h["round"] / rounds)
            status.info(f"Round {h['round']}/{rounds}: server → {n_cl} clients, local training, "
                        f"FedAvg done. MAE={h['mae']:.2f}, round traffic={fmt_bytes(h['total'])}")
            hist = ss.setdefault("_live", []); hist.append(h)
            chart.line_chart(pd.DataFrame(hist).set_index("round")[["mae", "rmse"]])
        ss["_live"] = []
        model, hist = run_federated(clients, tests, rounds, on_round=cb)
        # Reference: centralized linear model on the SAME clients' pooled data
        Xp = np.vstack([v[0] for v in clients.values()]); yp = np.concatenate([v[1] for v in clients.values()])
        Xt = np.vstack([v[0] for v in tests.values()]); yt = np.concatenate([v[1] for v in tests.values()])
        cl = LinearModel(); cl.fit(Xp, yp, epochs=40)
        ss["fed"] = dict(history=hist, clients=names, rounds=rounds,
                         cen_mae=mae(yt * SCALE, cl.predict(Xt) * SCALE),
                         cen_rmse=rmse(yt * SCALE, cl.predict(Xt) * SCALE),
                         raw_bytes=int(Xp.astype(np.float32).nbytes + yp.astype(np.float32).nbytes))
        status.success("Training complete ✅")
    fed = ss.get("fed")
    if fed:
        h = pd.DataFrame(fed["history"])
        st.plotly_chart(px.line(h, x="round", y=["mae", "rmse"], markers=True,
                                title="Federated model performance by round", **CHART), use_container_width=True)
        last = fed["history"][-1]
        cc = pd.DataFrame({"client": list(last["client_share"]),
                           "data share (FedAvg weight)": list(last["client_share"].values()),
                           "MAE of global model on client": list(last["client_mae"].values())})
        st.plotly_chart(px.bar(cc, x="client", y="data share (FedAvg weight)",
                               title="Client contribution to the global model", **CHART), use_container_width=True)
        st.dataframe(cc, hide_index=True)

elif page == "Communication Cost":
    fed = ss.get("fed")
    if not fed:
        st.warning("Run federated training first (Federated Learning page).")
    else:
        h = pd.DataFrame(fed["history"])
        st.metric("Model update size (per client, per direction)", fmt_bytes(params_bytes(np.zeros(10))))
        t = pd.DataFrame({"Round": h["round"], "Upload": h["upload"].map(fmt_bytes),
                          "Download": h["download"].map(fmt_bytes), "Total": h["total"].map(fmt_bytes),
                          "Per client": h["per_client"].map(fmt_bytes),
                          "Cumulative": h["cumulative"].map(fmt_bytes)})
        st.dataframe(t, hide_index=True, use_container_width=True)
        f = go.Figure()
        f.add_bar(x=h["round"], y=h["upload"], name="Upload"); f.add_bar(x=h["round"], y=h["download"], name="Download")
        f.add_scatter(x=h["round"], y=h["cumulative"], name="Cumulative", mode="lines+markers")
        f.update_layout(barmode="stack", title="Communication cost vs federated rounds (bytes)", **CHART)
        st.plotly_chart(f, use_container_width=True)
        st.caption("Measured from actual parameter vectors (float32). Upload = clients→server, download = server→clients.")

elif page == "Model Performance":
    fed = ss.get("fed")
    if not fed:
        st.warning("Run federated training first (Federated Learning page).")
    else:
        last = fed["history"][-1]
        comp = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "Training rounds", "Communication cost", "Data-sharing approach"],
            "Centralized ML": [f"{fed['cen_mae']:.2f}", f"{fed['cen_rmse']:.2f}", "1 (pooled training)",
                               f"{fmt_bytes(fed['raw_bytes'])} (raw data moved to server, one-time)",
                               "Raw data → central server"],
            "Federated ML": [f"{last['mae']:.2f}", f"{last['rmse']:.2f}", str(fed["rounds"]),
                             f"{fmt_bytes(last['cumulative'])} (model updates)",
                             "Raw data stays at clients; updates → server"]})
        st.dataframe(comp, hide_index=True, use_container_width=True)
        m = pd.DataFrame({"Approach": ["Centralized", "Federated"] * 2, "Metric": ["MAE"] * 2 + ["RMSE"] * 2,
                          "Value": [fed["cen_mae"], last["mae"], fed["cen_rmse"], last["rmse"]]})
        st.plotly_chart(px.bar(m, x="Metric", y="Value", color="Approach", barmode="group",
                               title="Centralized vs federated performance", **CHART), use_container_width=True)
        st.warning("Federated learning does NOT always cost less bandwidth. With many rounds or large models, "
                   "update traffic can exceed the raw data size. This tool measures it for your setup.")
        st.success(FINAL_MSG)

else:
    st.subheader("About SmartFlow FL")
    st.code("""                 GLOBAL SERVER (FedAvg)
                       |
        ---------------------------------
        |           |           |       |
   Junction A  Junction B  Junction C  Junction D
        |           |           |       |
    Local ML    Local ML    Local ML   Local ML
        |           |           |       |
        ------- model updates only ------""", language="text")
    a, b = st.columns(2)
    a.error("**Centralized**\n\nRaw traffic data → central server")
    b.success("**Federated**\n\nRaw traffic data stays at each client\n\nModel updates → server")
    st.caption("This demo shows the architecture. It is NOT a guarantee of complete privacy or security: "
               "model updates can leak information without extra protections (secure aggregation, differential privacy).")
    st.info("Clients are simulated on one computer; the data is synthetic unless you upload a CSV (PS-003).")
    st.success(FINAL_MSG)
