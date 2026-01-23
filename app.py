from __future__ import annotations

from datetime import datetime, date, time
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from supabase import create_client, Client


DEFAULT_INSTRUMENTS = ["Gold", "Silver"]
DIRECTIONS = ["Long", "Short"]
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]


TEXT: Dict[str, Dict[str, str]] = {
    "CZ": {
        "app_title": "Trading deník",
        "language": "Jazyk",
        "dark_mode": "Dark mode",
        "auth_title": "Přihlášení",
        "email": "Email",
        "password": "Heslo",
        "sign_in": "Přihlásit",
        "sign_up": "Registrovat",
        "sign_out": "Odhlásit",
        "signed_in_as": "Přihlášen jako",
        "please_login": "Přihlas se nebo se zaregistruj.",
        "confirm_email": "Pokud je vyžadováno potvrzení emailu, potvrď ho a pak se přihlas.",
        "tabs_add": "Přidat obchod",
        "tabs_stats": "Statistiky",
        "tabs_table": "Tabulka",
        "new_trade": "Nový obchod",
        "save_trade": "Uložit obchod",
        "saved": "Obchod uložen",
        "no_data": "Zatím nejsou data",
        "filters": "Filtry",
        "only_plan": "Jen obchody, kde byl dodržen plán",
        "export": "Export",
        "download_csv": "Stáhnout filtrovaná data jako CSV",
        "metric_trades": "Počet obchodů",
        "metric_winrate": "Winrate",
        "metric_avg_r": "Průměrné R",
        "metric_pf": "Profit factor",
        "equity_curve": "Equity křivka podle R",
        "best_setups": "Nejčastější setupy",
        "instrument": "Instrument",
        "instrument_mode": "Volba instrumentu",
        "instrument_pick": "Vybrat ze seznamu",
        "instrument_custom": "Vlastní instrument",
        "direction": "Směr",
        "trade_date": "Datum",
        "trade_time": "Čas",
        "entry": "Vstup",
        "stop_loss": "Stop Loss",
        "take_profit": "Take Profit",
        "exit": "Výstup",
        "risk_usd": "Risk v USD",
        "result_usd": "Výsledek v USD",
        "setup": "Setup",
        "timeframe": "Timeframe",
        "followed_plan": "Dodržen plán",
        "reason": "Důvod vstupu",
        "notes": "Poznámky",
        "delete_trade": "Smazat obchod",
        "select_trade": "Vyber obchod",
        "delete_btn": "Smazat",
        "delete_done": "Obchod smazán",
        "login_failed": "Přihlášení selhalo",
        "signup_failed": "Registrace selhala",
        "required_email": "Vyplň email",
        "required_password": "Vyplň heslo",
        "required_instrument": "Vyplň instrument",
        "risk_zero": "Risk nesmí být nula",
        "refresh": "Obnovit data",
    },
    "EN": {
        "app_title": "Trading Journal",
        "language": "Language",
        "dark_mode": "Dark mode",
        "auth_title": "Sign in",
        "email": "Email",
        "password": "Password",
        "sign_in": "Sign in",
        "sign_up": "Sign up",
        "sign_out": "Sign out",
        "signed_in_as": "Signed in as",
        "please_login": "Sign in or create an account.",
        "confirm_email": "If email confirmation is required, confirm your email and then sign in.",
        "tabs_add": "Add trade",
        "tabs_stats": "Stats",
        "tabs_table": "Table",
        "new_trade": "New trade",
        "save_trade": "Save trade",
        "saved": "Trade saved",
        "no_data": "No data yet",
        "filters": "Filters",
        "only_plan": "Only trades with plan followed",
        "export": "Export",
        "download_csv": "Download filtered CSV",
        "metric_trades": "Trades",
        "metric_winrate": "Win rate",
        "metric_avg_r": "Avg R",
        "metric_pf": "Profit factor",
        "equity_curve": "Equity curve by R",
        "best_setups": "Top setups",
        "instrument": "Instrument",
        "instrument_mode": "Instrument mode",
        "instrument_pick": "Pick from list",
        "instrument_custom": "Custom instrument",
        "direction": "Direction",
        "trade_date": "Date",
        "trade_time": "Time",
        "entry": "Entry",
        "stop_loss": "Stop Loss",
        "take_profit": "Take Profit",
        "exit": "Exit",
        "risk_usd": "Risk USD",
        "result_usd": "Result USD",
        "setup": "Setup",
        "timeframe": "Timeframe",
        "followed_plan": "Plan followed",
        "reason": "Reason",
        "notes": "Notes",
        "delete_trade": "Delete trade",
        "select_trade": "Select trade",
        "delete_btn": "Delete",
        "delete_done": "Trade deleted",
        "login_failed": "Login failed",
        "signup_failed": "Sign up failed",
        "required_email": "Email is required",
        "required_password": "Password is required",
        "required_instrument": "Instrument is required",
        "risk_zero": "Risk must not be zero",
        "refresh": "Refresh data",
    },
}


def t(key: str) -> str:
    lang = st.session_state.get("lang", "CZ")
    return TEXT.get(lang, TEXT["CZ"]).get(key, key)


def get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)


def set_theme(dark_mode: bool) -> str:
    if dark_mode:
        st.markdown(
            """
            <style>
            .stApp { background-color: #0e1117; color: #e6e6e6; }
            [data-testid="stSidebar"] { background-color: #111827; }
            [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
            [data-testid="stWidgetLabel"] p,
            [data-testid="stWidgetLabel"] label,
            [data-testid="stMarkdownContainer"] p { color: #cbd5e1 !important; }

            input, textarea {
                color: #e6e6e6 !important;
                background-color: #0b1220 !important;
                border: 1px solid #253047 !important;
            }

            [data-baseweb="select"] > div {
                background-color: #0b1220 !important;
                border: 1px solid #253047 !important;
            }
            [data-baseweb="select"] * { color: #e6e6e6 !important; }

            ul[role="listbox"]{
                background-color: #0b1220 !important;
                border: 1px solid #253047 !important;
            }
            ul[role="listbox"] li{ color: #e6e6e6 !important; }
            ul[role="listbox"] li:hover{ background-color: #111827 !important; }
            ul[role="listbox"] li[aria-selected="true"]{ background-color: #1f2937 !important; }

            button[role="tab"] { color: #e6e6e6 !important; }
            [data-testid="stCheckbox"] label { color: #e6e6e6 !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return "plotly_dark"
    else:
        st.markdown(
            """
            <style>
            .stApp { background-color: #ffffff; color: #111111; }
            [data-testid="stSidebar"] { background-color: #f3f4f6; }
            [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return "plotly_white"


def compute_result_r(result_usd: Optional[float], risk_usd: Optional[float]) -> Optional[float]:
    if result_usd is None or risk_usd is None:
        return None
    try:
        rsk = float(risk_usd)
        if rsk == 0:
            return None
        return float(result_usd) / rsk
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def ensure_session(supabase: Client) -> bool:
    sess = st.session_state.get("sb_session")
    if not sess:
        return False
    try:
        supabase.auth.set_session(sess["access_token"], sess["refresh_token"])
        return True
    except Exception:
        return False


@st.cache_data(ttl=10)
def fetch_trades_cached(_: str) -> pd.DataFrame:
    supabase = get_supabase()
    if not ensure_session(supabase):
        return pd.DataFrame()
    resp = supabase.table("trades").select("*").order("created_at", desc=True).execute()
    data = resp.data or []
    return pd.DataFrame(data)


def clear_trades_cache():
    fetch_trades_cached.clear()


def insert_trade(supabase: Client, payload: Dict[str, Any]) -> None:
    supabase.table("trades").insert(payload).execute()


def delete_trade(supabase: Client, trade_id: str) -> None:
    supabase.table("trades").delete().eq("id", trade_id).execute()


def calc_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"trades": 0, "winrate": None, "avg_r": None, "profit_factor": None}

    r = pd.to_numeric(df.get("result_r", pd.Series(dtype=float)), errors="coerce").dropna()
    if r.empty:
        return {"trades": int(len(df)), "winrate": None, "avg_r": None, "profit_factor": None}

    wins = r[r > 0]
    losses = r[r < 0]
    winrate = float(len(wins) / len(r)) if len(r) else None
    avg_r = float(r.mean()) if len(r) else None

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else None

    return {"trades": int(len(df)), "winrate": winrate, "avg_r": avg_r, "profit_factor": profit_factor}


st.set_page_config(page_title="Trading deník", layout="wide")

if "lang" not in st.session_state:
    st.session_state["lang"] = "CZ"

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

supabase = get_supabase()

with st.sidebar:
    st.session_state["lang"] = st.selectbox(t("language"), ["CZ", "EN"], index=0 if st.session_state["lang"] == "CZ" else 1)
    st.session_state["dark_mode"] = st.toggle(t("dark_mode"), value=st.session_state["dark_mode"])

plotly_template = set_theme(st.session_state["dark_mode"])

st.title(t("app_title"))

authed = ensure_session(supabase)

with st.sidebar:
    st.subheader(t("auth_title"))

    if authed:
        email = st.session_state.get("sb_user_email", "")
        st.caption(f"{t('signed_in_as')} {email}")
        if st.button(t("sign_out")):
            st.session_state.pop("sb_session", None)
            st.session_state.pop("sb_user_email", None)
            clear_trades_cache()
            st.rerun()
    else:
        email = st.text_input(t("email"))
        password = st.text_input(t("password"), type="password")

        c1, c2 = st.columns(2)

        with c1:
            if st.button(t("sign_in")):
                if not email:
                    st.warning(t("required_email"))
                elif not password:
                    st.warning(t("required_password"))
                else:
                    try:
                        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        session = res.session
                        if session:
                            st.session_state["sb_session"] = {
                                "access_token": session.access_token,
                                "refresh_token": session.refresh_token,
                            }
                            st.session_state["sb_user_email"] = email
                            clear_trades_cache()
                            st.rerun()
                        else:
                            st.error(t("login_failed"))
                    except Exception as e:
                        st.error(f"{t('login_failed')}: {e}")

        with c2:
            if st.button(t("sign_up")):
                if not email:
                    st.warning(t("required_email"))
                elif not password:
                    st.warning(t("required_password"))
                else:
                    try:
                        res = supabase.auth.sign_up({"email": email, "password": password})
                        if res.user:
                            st.success(t("confirm_email"))
                        else:
                            st.error(t("signup_failed"))
                    except Exception as e:
                        st.error(f"{t('signup_failed')}: {e}")

if not authed:
    st.info(t("please_login"))
    st.stop()

session_key = st.session_state["sb_session"]["access_token"]

c_top1, c_top2 = st.columns([1, 1])
with c_top1:
    if st.button(t("refresh")):
        clear_trades_cache()
        st.rerun()

df = fetch_trades_cached(session_key)

with st.sidebar:
    st.divider()
    st.header(t("filters"))
    instrument_filter = st.multiselect(t("instrument"), DEFAULT_INSTRUMENTS, default=DEFAULT_INSTRUMENTS)
    direction_filter = st.multiselect(t("direction"), DIRECTIONS, default=DIRECTIONS)
    only_plan = st.checkbox(t("only_plan"), value=False)

work = df.copy()
if not work.empty:
    if "instrument" in work.columns:
        work = work[work["instrument"].isin(instrument_filter)]
    if "direction" in work.columns:
        work = work[work["direction"].isin(direction_filter)]
    if only_plan and "followed_plan" in work.columns:
        work = work[work["followed_plan"].astype(str).str.lower().isin(["true", "1", "yes", "ano"])]

tab_add, tab_stats, tab_table = st.tabs([t("tabs_add"), t("tabs_stats"), t("tabs_table")])


with tab_add:
    st.subheader(t("new_trade"))

    now = datetime.now()
    default_date = now.date()
    default_time = now.time().replace(second=0, microsecond=0)

    with st.form("add_trade_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            trade_date = st.date_input(t("trade_date"), value=default_date)
            trade_time = st.time_input(t("trade_time"), value=default_time)

            instr_mode = st.radio(
                t("instrument_mode"),
                [t("instrument_pick"), t("instrument_custom")],
                horizontal=True
            )

            if instr_mode == t("instrument_custom"):
                instrument = st.text_input(t("instrument"), placeholder="Např. EURUSD, BTC, SP500")
            else:
                instrument = st.selectbox(t("instrument"), DEFAULT_INSTRUMENTS)

            direction = st.selectbox(t("direction"), DIRECTIONS)

        with c2:
            entry = st.number_input(t("entry"), min_value=0.0, value=0.0, step=0.01, format="%.2f")
            stop_loss = st.number_input(t("stop_loss"), min_value=0.0, value=0.0, step=0.01, format="%.2f")
            take_profit = st.number_input(t("take_profit"), min_value=0.0, value=0.0, step=0.01, format="%.2f")
            exit_price = st.number_input(t("exit"), min_value=0.0, value=0.0, step=0.01, format="%.2f")

        with c3:
            risk_usd = st.number_input(t("risk_usd"), min_value=0.0, value=0.0, step=1.0, format="%.2f")
            result_usd = st.number_input(t("result_usd"), value=0.0, step=1.0, format="%.2f")
            setup = st.text_input(t("setup"))
            timeframe = st.selectbox(t("timeframe"), TIMEFRAMES, index=2)
            followed_plan = st.checkbox(t("followed_plan"), value=True)

        reason = st.text_area(t("reason"))
        notes = st.text_area(t("notes"))

        submitted = st.form_submit_button(t("save_trade"))

    if submitted:
        instrument_clean = (instrument or "").strip()
        if not instrument_clean:
            st.warning(t("required_instrument"))
        elif safe_float(risk_usd) == 0:
            st.warning(t("risk_zero"))
        else:
            risk_val = safe_float(risk_usd)
            result_val = safe_float(result_usd)
            r_val = compute_result_r(result_val, risk_val)

            payload = {
                "trade_date": str(trade_date) if isinstance(trade_date, date) else None,
                "trade_time": trade_time.strftime("%H:%M:%S") if isinstance(trade_time, time) else None,
                "instrument": instrument_clean,
                "direction": direction,
                "entry": entry if entry > 0 else None,
                "stop_loss": stop_loss if stop_loss > 0 else None,
                "take_profit": take_profit if take_profit > 0 else None,
                "exit": exit_price if exit_price > 0 else None,
                "risk_usd": risk_val if (risk_val is not None and risk_val > 0) else None,
                "result_usd": result_val,
                "result_r": r_val,
                "setup": (setup or "").strip(),
                "timeframe": timeframe,
                "reason": (reason or "").strip(),
                "followed_plan": bool(followed_plan),
                "notes": (notes or "").strip(),
            }

            insert_trade(supabase, payload)
            clear_trades_cache()
            st.success(t("saved"))
            st.rerun()


with tab_stats:
    m = calc_metrics(work)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("metric_trades"), f"{m['trades']}")
    c2.metric(t("metric_winrate"), f"{(m['winrate'] * 100):.1f} %" if m["winrate"] is not None else "N/A")
    c3.metric(t("metric_avg_r"), f"{m['avg_r']:.2f}" if m["avg_r"] is not None else "N/A")
    c4.metric(t("metric_pf"), f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else "N/A")

    st.divider()
    st.subheader(t("equity_curve"))

    if work.empty or "result_r" not in work.columns:
        st.info(t("no_data"))
    else:
        tmp = work.copy()
        tmp["result_r"] = pd.to_numeric(tmp["result_r"], errors="coerce").fillna(0)

        if "created_at" in tmp.columns:
            tmp = tmp.sort_values("created_at")

        tmp["cum_r"] = tmp["result_r"].cumsum()
        tmp["idx"] = range(len(tmp))

        fig = px.line(tmp, x="idx", y="cum_r", template=plotly_template)
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader(t("best_setups"))

    if work.empty or "setup" not in work.columns:
        st.info(t("no_data"))
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
    if work.empty:
        st.info(t("no_data"))
    else:
        show_cols = [
            "created_at",
            "instrument",
            "direction",
            "result_usd",
            "result_r",
            "setup",
            "timeframe",
            "followed_plan",
        ]
        existing = [c for c in show_cols if c in work.columns]
        st.dataframe(work[existing], use_container_width=True)

        st.divider()
        st.subheader(t("export"))
        csv = work.to_csv(index=False).encode("utf-8")
        st.download_button(t("download_csv"), data=csv, file_name="trades_filtered.csv", mime="text/csv")

        st.divider()
        st.subheader(t("delete_trade"))

        ids = work["id"].astype(str).tolist() if "id" in work.columns else []
        selected_id = st.selectbox(t("select_trade"), options=[""] + ids)

        if selected_id:
            if st.button(t("delete_btn")):
                delete_trade(supabase, selected_id)
                clear_trades_cache()
                st.success(t("delete_done"))
                st.rerun()
