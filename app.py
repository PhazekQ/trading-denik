import os
from datetime import datetime

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px


DATA_FILE = "trades.csv"

COLUMNS = [
    "timestamp_added",
    "trade_date",
    "trade_time",
    "instrument",
    "direction",
    "entry",
    "stop_loss",
    "take_profit",
    "exit",
    "risk_usd",
    "result_usd",
    "result_r",
    "setup",
    "timeframe",
    "reason",
    "followed_plan",
    "notes",
]

INSTRUMENTS = ["Gold", "Silver"]
DIRECTIONS = ["Long", "Short"]
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]


def ensure_data_file():
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=COLUMNS).to_csv(DATA_FILE, index=False)
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        if content == "":
            pd.DataFrame(columns=COLUMNS).to_csv(DATA_FILE, index=False)
    except Exception:
        pd.DataFrame(columns=COLUMNS).to_csv(DATA_FILE, index=False)


def load_data() -> pd.DataFrame:
    ensure_data_file()

    try:
        df = pd.read_csv(DATA_FILE)
    except Exception:
        pd.DataFrame(columns=COLUMNS).to_csv(DATA_FILE, index=False)
        df = pd.DataFrame(columns=COLUMNS)

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[COLUMNS]

    for col in ["entry", "stop_loss", "take_profit", "exit", "risk_usd", "result_usd", "result_r"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def append_trade(row: dict):
    df = load_data()
    df2 = pd.DataFrame([row], columns=COLUMNS)
    out = pd.concat([df, df2], ignore_index=True)
    out.to_csv(DATA_FILE, index=False)


def compute_result_r(result_usd: float, risk_usd: float) -> float:
    if risk_usd is None or pd.isna(risk_usd) or float(risk_usd) == 0:
        return float("nan")
    if result_usd is None or pd.isna(result_usd):
        return float("nan")
    return float(result_usd) / float(risk_usd)


def equity_curve_r(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()

    work["dt"] = pd.to_datetime(
        work["trade_date"].astype(str).fillna("") + " " + work["trade_time"].astype(str).fillna(""),
        errors="coerce",
    )

    if work["dt"].notna().any():
        work = work.sort_values("dt")
    else:
        work = work.reset_index(drop=True)

    work["cum_r"] = work["result_r"].fillna(0).cumsum()
    return work


def metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "trades": 0,
            "winrate": float("nan"),
            "avg_r": float("nan"),
            "avg_win_r": float("nan"),
            "avg_loss_r": float("nan"),
            "profit_factor": float("nan"),
        }

    r = df["result_r"].dropna()
    trades = int(len(df))

    if r.empty:
        return {
            "trades": trades,
            "winrate": float("nan"),
            "avg_r": float("nan"),
            "avg_win_r": float("nan"),
            "avg_loss_r": float("nan"),
            "profit_factor": float("nan"),
        }

    wins = r[r > 0]
    losses = r[r < 0]

    winrate = float(len(wins) / len(r)) if len(r) else float("nan")
    avg_r = float(r.mean()) if len(r) else float("nan")
    avg_win_r = float(wins.mean()) if len(wins) else float("nan")
    avg_loss_r = float(losses.mean()) if len(losses) else float("nan")

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("nan")

    return {
        "trades": trades,
        "winrate": winrate,
        "avg_r": avg_r,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "profit_factor": profit_factor,
    }


@st.cache_data(ttl=30)
def fetch_intraday(ticker: str, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Time"}, inplace=True)
    elif "Date" in df.columns:
        df.rename(columns={"Date": "Time"}, inplace=True)

    # yfinance občas vrací MultiIndex sloupce
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # pojistka: kdyby zůstaly tuple názvy
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    return df


# ---------------- UI ----------------
st.set_page_config(page_title="Trading deník", layout="wide")
st.title("Trading deník")

df = load_data()

# Theme state default
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True


# Sidebar: theme + filters together (čisté a přehledné)
with st.sidebar:
    st.session_state["dark_mode"] = st.toggle("Dark mode", value=st.session_state["dark_mode"])
    st.divider()

    st.header("Filtry")
    instrument_filter = st.multiselect("Instrument", INSTRUMENTS, default=INSTRUMENTS)
    direction_filter = st.multiselect("Směr", DIRECTIONS, default=DIRECTIONS)
    only_plan = st.checkbox("Jen obchody, kde byl dodržen plán", value=False)

    st.divider()
    st.caption("Data se ukládají do trades.csv ve stejné složce jako app.py")


# CSS theme + Plotly theme
if st.session_state["dark_mode"]:
    plotly_template = "plotly_dark"
    st.markdown(
        """
        <style>
        /* Základ */
        .stApp { background-color: #0e1117; color: #e6e6e6; }
        [data-testid="stSidebar"] { background-color: #111827; }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }

        /* Texty */
        h1, h2, h3, h4, h5, h6 { color: #e6e6e6 !important; }
        [data-testid="stMarkdownContainer"] p { color: #cbd5e1 !important; }

        /* Labely u widgetů */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label { color: #cbd5e1 !important; }

        /* Inputy */
        input, textarea {
            color: #e6e6e6 !important;
            background-color: #0b1220 !important;
            border: 1px solid #253047 !important;
        }

        /* Selectbox / Multiselect - control */
        [data-baseweb="select"] > div {
            background-color: #0b1220 !important;
            border: 1px solid #253047 !important;
        }
        [data-baseweb="select"] * { color: #e6e6e6 !important; }

        /* Dropdown menu (options list) */
        ul[role="listbox"]{
            background-color: #0b1220 !important;
            border: 1px solid #253047 !important;
        }
        ul[role="listbox"] li{
            color: #e6e6e6 !important;
        }
        ul[role="listbox"] li:hover{
            background-color: #111827 !important;
        }
        ul[role="listbox"] li[aria-selected="true"]{
            background-color: #1f2937 !important;
        }

        /* Taby */
        button[role="tab"] { color: #e6e6e6 !important; }

        /* Checkbox text */
        [data-testid="stCheckbox"] label { color: #e6e6e6 !important; }

        /* Metriky */
        [data-testid="stMetricLabel"] { color: #cbd5e1 !important; }

        /* Linky */
        a { color: #93c5fd !important; }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    plotly_template = "plotly_white"
    st.markdown(
        """
        <style>
        .stApp { background-color: #ffffff; color: #111111; }
        [data-testid="stSidebar"] { background-color: #f3f4f6; }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
        </style>
        """,
        unsafe_allow_html=True
    )


work = df.copy()
if not work.empty:
    work["instrument"] = work["instrument"].astype(str)
    work["direction"] = work["direction"].astype(str)

    work = work[work["instrument"].isin(instrument_filter)]
    work = work[work["direction"].isin(direction_filter)]

    if only_plan:
        work = work[work["followed_plan"].astype(str).str.lower().isin(["true", "1", "yes", "ano"])]

tab_add, tab_stats, tab_table, tab_live = st.tabs(["Přidat obchod", "Statistiky", "Tabulka", "Trh live"])


with tab_add:
    st.subheader("Nový obchod")

    now = datetime.now()
    default_date = now.date().isoformat()
    default_time = now.strftime("%H:%M")

    with st.form("add_trade_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            trade_date = st.text_input("Datum (YYYY-MM-DD)", value=default_date)
            trade_time = st.text_input("Čas (HH:MM)", value=default_time)
            instrument = st.selectbox("Instrument", INSTRUMENTS)
            direction = st.selectbox("Směr", DIRECTIONS)

        with c2:
            entry = st.number_input("Vstup", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            stop_loss = st.number_input("Stop Loss", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            take_profit = st.number_input("Take Profit", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            exit_price = st.number_input("Výstup (pokud už uzavřeno)", min_value=0.0, value=0.0, step=0.01, format="%.2f")

        with c3:
            risk_usd = st.number_input("Risk v USD", min_value=0.0, value=0.0, step=1.0, format="%.2f")
            result_usd = st.number_input("Výsledek v USD", value=0.0, step=1.0, format="%.2f")
            setup = st.text_input("Setup (např. breakout, pullback)")
            timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=2)
            followed_plan = st.checkbox("Dodržen plán", value=True)

        reason = st.text_area("Důvod vstupu (jedna věta)")
        notes = st.text_area("Poznámky (volitelné)")

        submitted = st.form_submit_button("Uložit obchod")

    if submitted:
        ts_added = datetime.now().isoformat(timespec="seconds")
        result_r = compute_result_r(result_usd, risk_usd)

        row = {
            "timestamp_added": ts_added,
            "trade_date": trade_date.strip(),
            "trade_time": trade_time.strip(),
            "instrument": instrument,
            "direction": direction,
            "entry": entry if entry > 0 else None,
            "stop_loss": stop_loss if stop_loss > 0 else None,
            "take_profit": take_profit if take_profit > 0 else None,
            "exit": exit_price if exit_price > 0 else None,
            "risk_usd": risk_usd if risk_usd > 0 else None,
            "result_usd": result_usd,
            "result_r": result_r,
            "setup": setup.strip(),
            "timeframe": timeframe,
            "reason": reason.strip(),
            "followed_plan": bool(followed_plan),
            "notes": notes.strip(),
        }

        append_trade(row)
        st.success("Obchod uložen. Přepni na Statistiky nebo Tabulka.")


with tab_stats:
    st.subheader("Statistiky")
    m = metrics(work)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Počet obchodů", f"{m['trades']}")
    c2.metric("Winrate", f"{(m['winrate'] * 100):.1f} %" if pd.notna(m["winrate"]) else "N/A")
    c3.metric("Průměrné R", f"{m['avg_r']:.2f}" if pd.notna(m["avg_r"]) else "N/A")
    c4.metric("Průměrná výhra R", f"{m['avg_win_r']:.2f}" if pd.notna(m["avg_win_r"]) else "N/A")
    c5.metric("Profit factor", f"{m['profit_factor']:.2f}" if pd.notna(m["profit_factor"]) else "N/A")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("### Equity křivka podle R")
        eq = equity_curve_r(work)
        if eq.empty or "cum_r" not in eq.columns:
            st.info("Zatím nejsou data pro křivku.")
        else:
            fig = plt.figure()
            plt.plot(eq["cum_r"].values)
            plt.xlabel("Obchody")
            plt.ylabel("Kumulativní R")
            st.pyplot(fig, clear_figure=True)

    with right:
        st.markdown("### Výkon podle instrumentu")
        if work.empty:
            st.info("Zatím nejsou data.")
        else:
            g = work.groupby("instrument", dropna=False)["result_r"].mean().reset_index()
            g.columns = ["Instrument", "Průměrné R"]
            st.dataframe(g, use_container_width=True)

    st.divider()
    st.markdown("### Nejčastější setupy")
    if work.empty:
        st.info("Zatím nejsou data.")
    else:
        s = (
            work.assign(setup_clean=work["setup"].fillna("").astype(str).str.strip())
            .query("setup_clean != ''")
            .groupby("setup_clean")["result_r"]
            .agg(["count", "mean"])
            .reset_index()
            .sort_values(["count", "mean"], ascending=[False, False])
        )
        s.columns = ["Setup", "Počet", "Průměrné R"]
        st.dataframe(s, use_container_width=True)


with tab_table:
    st.subheader("Tabulka obchodů")
    try:
        table_view = work.sort_values(["trade_date", "trade_time"], ascending=[False, False])
    except Exception:
        table_view = work

    st.dataframe(table_view, use_container_width=True)

    st.divider()
    st.markdown("### Export")
    if st.button("Stáhnout filtrovaná data jako CSV"):
        csv = work.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Klikni pro stažení CSV",
            data=csv,
            file_name="trades_filtered.csv",
            mime="text/csv",
        )


with tab_live:
    st.subheader("Trh live (přehled)")

    mode = st.radio(
        "Zobrazení",
        ["Last", "1m", "5m", "15m", "60m"],
        horizontal=True
    )

    if st.button("Refresh"):
        st.cache_data.clear()

    interval_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "60m": "60m",
    }
    interval = interval_map.get(mode, "1m")

    colA, colB = st.columns(2)

    # --- GOLD ---
    with colA:
        st.markdown("### Gold (GC=F)")
        gold = fetch_intraday("GC=F", interval=interval)

        if gold.empty:
            st.info("Nepodařilo se načíst data pro Gold.")
        else:
            last_price = float(gold["Close"].dropna().iloc[-1])
            st.metric("Poslední cena", f"{last_price:.2f}")

            if mode == "Last":
                data_to_plot = gold.tail(20)
            else:
                data_to_plot = gold

            fig = px.line(data_to_plot, x="Time", y="Close", template=plotly_template)
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=260,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Najetím myší uvidíš přesné hodnoty. Data mohou být zpožděná.")

    # --- SILVER ---
    with colB:
        st.markdown("### Silver (SI=F)")
        silver = fetch_intraday("SI=F", interval=interval)

        if silver.empty:
            st.info("Nepodařilo se načíst data pro Silver.")
        else:
            last_price = float(silver["Close"].dropna().iloc[-1])
            st.metric("Poslední cena", f"{last_price:.2f}")

            if mode == "Last":
                data_to_plot = silver.tail(20)
            else:
                data_to_plot = silver

            fig = px.line(data_to_plot, x="Time", y="Close", template=plotly_template)
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=260,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Najetím myší uvidíš přesné hodnoty. Data mohou být zpožděná.")

    st.caption("Poznámka: Data jsou z Yahoo Finance a mohou být zpožděná.")