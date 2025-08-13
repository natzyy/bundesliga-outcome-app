import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt
import os, requests, re, unicodedata, datetime as dt
from pathlib import Path

# -------------------------------------------------
# Seite konfigurieren
# -------------------------------------------------
st.set_page_config(page_title="Bundesliga Spielergebnis", page_icon="⚽", layout="centered")
st.sidebar.success("Build: UI-Mode + CSV/LiveOdds (combined)")

# -------------------------------------------------
# Daten laden (historische CSVs für Quoten-Autofill)
# -------------------------------------------------
@st.cache_data
def load_matches(paths):
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="latin-1")
        df["source_file"] = Path(p).name
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    # Teams müssen vorhanden sein; Quoten dürfen fehlen
    keep_cols = [c for c in ["HomeTeam","AwayTeam"] if c in df_all.columns]
    if keep_cols:
        df_all = df_all.dropna(subset=keep_cols)

    # Quoten in float konvertieren
    for col in ["B365H","B365D","B365A","PSH","PSD","PSA"]:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    # Datum parsen (falls vorhanden)
    if "Date" in df_all.columns:
        df_all["Date"] = pd.to_datetime(df_all["Date"], dayfirst=True, errors="coerce")
    return df_all

DATA_PATHS = [
    "data/D1_2020_2021.csv",
    "data/D1_2021_2022.csv",
    "data/D1_2022_2023.csv",
    "data/D1_2023_2024.csv",
    "data/D1_2024_2025.csv",
]
try:
    df_all = load_matches(DATA_PATHS)
except Exception as e:
    st.warning(f"Konnte CSVs nicht laden: {e}")
    df_all = pd.DataFrame(columns=["HomeTeam","AwayTeam","B365H","B365D","B365A","PSH","PSD","PSA","Date"])

# -------------------------------------------------
# Normalisierung / Aliase
# -------------------------------------------------
def _norm(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(fc|sv|tsg|vfl|sc|borussia|bayer|1|04|05|1899|98|04)\b", " ", s)
    s = " ".join(s.split())
    return s

ALIASES = {
    "bayern munich": ["bayern munich", "bayern munchen", "fc bayern munchen", "bayern"],
    "augsburg": ["augsburg", "fc augsburg"],
    "borussia dortmund": ["borussia dortmund", "dortmund"],
    "rb leipzig": ["rb leipzig", "leipzig", "rasenballsport leipzig"],
    "bayer leverkusen": ["bayer leverkusen", "leverkusen"],
    "borussia monchengladbach": ["borussia monchengladbach", "monchengladbach", "gladbach"],
    "koln": ["koln", "fc koln", "1 fc koln", "koln 1 fc", "kolner"],
    "union berlin": ["union berlin", "1 fc union berlin", "union"],
    "eintracht frankfurt": ["eintracht frankfurt", "frankfurt", "ein frankfurt"],
    "werder bremen": ["werder bremen", "bremen"],
    "vfb stuttgart": ["stuttgart", "vfb stuttgart"],
    "vfl wolfsburg": ["wolfsburg"],
    "sc freiburg": ["freiburg"],
    "mainz": ["mainz", "mainz 05"],
    "tsg hoffenheim": ["hoffenheim"],
    "vfl bochum": ["bochum"],
    "heidenheim": ["heidenheim", "1 fc heidenheim"],
    "darmstadt": ["darmstadt", "sv darmstadt"],
    "st pauli": ["st pauli", "fc st pauli"],
    "fortuna dusseldorf": ["fortuna dusseldorf", "fortuna"],
    "hamburger sv": ["hamburger sv", "hamburg"],
}
def _alias_set(name: str) -> set:
    k = _norm(name)
    return set([k] + ALIASES.get(k, []))

def _any_match(user_set: set, ev_name: str) -> bool:
    evn = _norm(ev_name or "")
    if evn in user_set:
        return True
    return any((u in evn) or (evn in u) for u in user_set)

# -------------------------------------------------
# Provider A: The Odds API (v4) – H2H/3-Way
# -------------------------------------------------
@st.cache_data(ttl=120)
def _fetch_odds_oddsapi(home_team, away_team, sport_key="soccer_germany_bundesliga",
                        want_books=("pinnacle","bet365")):
    api_key = st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY","")).strip()
    if not api_key:
        raise RuntimeError("ODDS_API_KEY fehlt. In Streamlit → Settings → Secrets eintragen.")

    nh, na = _alias_set(home_team), _alias_set(away_team)

    def call(bookmakers_csv: str | None):
        params = {
            "apiKey": api_key,
            "regions": "eu,uk,us",   # bewusst breit
            "oddsFormat": "decimal",
            "markets": "h2h",
        }
        if bookmakers_csv:
            params["bookmakers"] = bookmakers_csv
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        return r.json()

    def find_match(items):
        for ev in items:
            h = ev.get("home_team","")
            a = ev.get("away_team","")
            if (_any_match(nh,h) and _any_match(na,a)) or (_any_match(nh,a) and _any_match(na,h)):
                flip = (_any_match(nh,a) and _any_match(na,h))
                return ev, flip
        return None, False

    def merge_event(ev, res: dict):
        for bk in ev.get("bookmakers", []):
            bname = (bk.get("key","") or "").lower()
            if bname not in want_books:
                continue
            for m in bk.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for out in m.get("outcomes", []):
                    nm = _norm(out.get("name",""))
                    price = float(out.get("price"))
                    code = None
                    if nm in ("draw","unentschieden"):
                        code = "D"
                    elif nm in nh:
                        code = "H"
                    elif nm in na:
                        code = "A"
                    elif nm in ("home","heim"):
                        code = "H"
                    elif nm in ("away","auswaerts","auswärts","auswarts","gast","away team"):
                        code = "A"
                    if not code:
                        continue
                    if bname == "pinnacle":
                        res[f"PS{code}"] = price
                    elif bname == "bet365":
                        res[f"B365{code}"] = price

    # 1) Kombiniert
    data = call(",".join(want_books))
    ev, _ = find_match(data)
    # 2) ohne Filter, falls nicht gefunden
    if not ev:
        data_all = call(None)
        ev, _ = find_match(data_all)
        if not ev:
            raise LookupError("Kein passendes Event in The Odds API gefunden.")

    res = {}
    merge_event(ev, res)

    # 3) je Bookie gezielt nachladen, falls im Kombi-Call fehlend
    need_b365 = any(k not in res for k in ("B365H","B365D","B365A"))
    need_ps   = any(k not in res for k in ("PSH","PSD","PSA"))
    if need_b365:
        ev_b, _ = find_match(call("bet365"))
        if ev_b: merge_event(ev_b, res)
    if need_ps:
        ev_p, _ = find_match(call("pinnacle"))
        if ev_p: merge_event(ev_p, res)

    need = ["B365H","B365D","B365A","PSH","PSD","PSA"]
    missing = [k for k in need if k not in res]
    return res, missing

# -------------------------------------------------
# Provider B: SportMonks – nur Bet365 ergänzen (Bookmaker 2, Market 1=1X2)
# -------------------------------------------------
@st.cache_data(ttl=120)
def _fetch_bet365_from_sportmonks(home_team, away_team, days_ahead=21):
    token = st.secrets.get("SPORTMONKS_TOKEN", os.getenv("SPORTMONKS_TOKEN","")).strip()
    if not token:
        return {}  # kein Token => kein Fallback

    nh, na = _alias_set(home_team), _alias_set(away_team)

    since = (dt.datetime.utcnow() - dt.timedelta(hours=12)).strftime("%Y-%m-%d")
    until = (dt.datetime.utcnow() + dt.timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # Fixtures im Fenster inkl. Teilnehmer
    fx_url = f"https://api.sportmonks.com/v3/football/fixtures/date/{since}/{until}"
    fx_params = {"api_token": token, "include": "participants"}
    fx = requests.get(fx_url, params=fx_params, timeout=12)
    fx.raise_for_status()
    fx_data = fx.json().get("data", []) or []

    fixture_id = None
    for ev in fx_data:
        names = [p.get("name","") for p in (ev.get("participants") or [])]
        if len(names) >= 2:
            if (_any_match(nh, names[0]) and _any_match(na, names[1])) or (_any_match(nh, names[1]) and _any_match(na, names[0])):
                fixture_id = ev.get("id")
                break
    if not fixture_id:
        return {}

    # Odds für Bet365 (Bookmaker 2) im Market 1 (1X2)
    odd_url = f"https://api.sportmonks.com/v3/odds/pre-match/fixtures/{fixture_id}/bookmakers/2"
    odd_params = {"api_token": token, "markets": "1"}
    od = requests.get(odd_url, params=odd_params, timeout=12)
    od.raise_for_status()
    odds_data = od.json().get("data", []) or []

    out = {}
    for market in odds_data:
        if str(market.get("market_id")) != "1":
            continue
        for o in market.get("odds", []) or []:
            nm = _norm(o.get("label","") or o.get("name",""))
            price = o.get("value") or o.get("decimal") or o.get("price")
            if price is None:
                continue
            price = float(price)
            if nm in ("home","heim","heimsieg"):
                out["B365H"] = price
            elif nm in ("draw","unentschieden"):
                out["B365D"] = price
            elif nm in ("away","auswaerts","auswärts","gast"):
                out["B365A"] = price
    return out

# -------------------------------------------------
# Kombi-Fetch: zuerst OddsAPI, fehlendes Bet365 via SportMonks ergänzen
# -------------------------------------------------
def fetch_live_odds_combined(home_team, away_team):
    note = None
    try:
        odds, missing = _fetch_odds_oddsapi(home_team, away_team)
    except Exception as e:
        odds, missing = {}, []
        note = f"Odds API nicht verfügbar: {e}"

    # nur Bet365 (falls fehlt) mit SportMonks ergänzen
    need_b365 = any(k not in odds for k in ("B365H","B365D","B365A"))
    if need_b365:
        try:
            b365 = _fetch_bet365_from_sportmonks(home_team, away_team)
            for k, v in b365.items():
                odds.setdefault(k, v)
        except Exception as e:
            note = (note + " | " if note else "") + f"SportMonks nicht verfügbar: {e}"

    need = ["B365H","B365D","B365A","PSH","PSD","PSA"]
    missing = [k for k in need if k not in odds]
    return odds, missing, note

# -------------------------------------------------
# Modell laden + Teamliste
# -------------------------------------------------
model = joblib.load("bundesliga_best_model.joblib")
try:
    ohe = model.named_steps["preprocess"].named_transformers_["cat"]
    home_teams = list(ohe.categories_[0])
    away_teams = list(ohe.categories_[1])
    teams = sorted(set(home_teams) | set(away_teams))
except Exception:
    teams = None  # Fallback

# -------------------------------------------------
# UI: Titel + Moduswahl
# -------------------------------------------------
st.title("⚽ Bundesliga Spielergebnis – Vorhersage")

has_history = not df_all.empty
choices = ["Zukünftiges Spiel"] + (["Historisches Spiel"] if has_history else [])
mode = st.radio(
    "Vorhersage-Modus",
    choices,
    index=0,
    help="Historischer Modus lädt bequem die damaligen Quoten. Die Vorhersage basiert IMMER auf einem Modell, das auf vielen vergangenen Jahren trainiert wurde."
)

# Trainingszeitraum anzeigen
train_span = ""
if "Date" in df_all.columns and df_all["Date"].notna().any():
    dmin, dmax = df_all["Date"].min(), df_all["Date"].max()
    if pd.notna(dmin) and pd.notna(dmax):
        train_span = f"Trainiert auf historischen Spielen ca. {dmin.date()} bis {dmax.date()}."
st.info(
    "ℹ️ **Hinweis**\n\n"
    "**Zukünftiges Spiel = echte Prognose:** Vorhersage basiert auf aktuell geladenen oder manuell eingegebenen Quoten.\n\n"
    "**Historisches Spiel = Backtest:** Damalige Quoten laden; Ergebnis war bekannt – wir prüfen nur, wie gut das Modell getroffen **hätte**."
    + (f"\n\n{train_span}" if train_span else "")
)

# -------------------------
# Eingaben (Teams)
# -------------------------
colA, colB = st.columns(2)
with colA:
    if teams:
        home_team = st.selectbox("Heimmannschaft", teams,
                                 index=teams.index("Bayern Munich") if "Bayern Munich" in teams else 0,
                                 key="home_team_select")
    else:
        home_team = st.text_input("Heimmannschaft", "Bayern Munich")
with colB:
    if teams:
        away_team = st.selectbox("Auswärtsmannschaft", teams,
                                 index=teams.index("Borussia Dortmund") if "Borussia Dortmund" in teams else 0,
                                 key="away_team_select")
    else:
        away_team = st.text_input("Auswärtsmannschaft", "Borussia Dortmund")

# -------------------------
# Live-Quoten (nur im Zukunfts-Modus)
# -------------------------
if mode.startswith("Zukünftiges"):
    if st.button("🔄 Live-Quoten laden (Bet365 & Pinnacle)"):
        try:
            odds, missing, provider_note = fetch_live_odds_combined(home_team, away_team)

            # Reset Hinweise
            st.session_state["odds_note"] = ""
            st.session_state["odds_origin"] = None

            # Nur echte, gelieferte Werte setzen (kein Spiegeln)
            for k, v in odds.items():
                st.session_state[k] = float(v)

            # Wenn Bet365 trotz beider Feeds fehlt → deutlicher Hinweis
            still_missing_b365 = any(k not in odds for k in ("B365H","B365D","B365A"))
            if still_missing_b365:
                st.warning("**Daten von Bet365 fehlen (Feed/Lizenz). "
                           "Bitte Bet365-Quoten manuell eintragen.**")

            if provider_note:
                st.info(provider_note)

            if not odds:
                st.error("Keine Live-Quoten gefunden – bitte manuell eintragen.")
            else:
                st.success("Live-Quoten geladen.")
        except Exception as e:
            st.error(f"Live-Quoten konnten nicht geladen werden: {e}")

# -------------------------
# Historischer CSV-Lookup (nur im Historik-Modus)
# -------------------------
if mode.startswith("Historisches"):
    cands = df_all[(df_all["HomeTeam"] == home_team) & (df_all["AwayTeam"] == away_team)].copy()
    selected_row = None
    if not cands.empty:
        if "Date" in cands.columns and cands["Date"].notna().any():
            cands = cands.sort_values("Date", ascending=False)
            date_choices = [d.strftime("%Y-%m-%d") if pd.notna(d) else "ohne Datum" for d in cands["Date"]]
            picked_date = st.selectbox("Historisches Spiel wählen", date_choices, key="picked_date")
            selected_row = cands.iloc[date_choices.index(picked_date)]
        else:
            selected_row = cands.iloc[0]
        if st.button("🔁 Quoten aus CSV übernehmen"):
            for k in ["B365H","B365D","B365A","PSH","PSD","PSA"]:
                if k in selected_row and not pd.isna(selected_row[k]):
                    st.session_state[k] = float(selected_row[k])
            st.toast("Quoten aus CSV übernommen.", icon="✅")
    else:
        st.info("Kein historisches Spiel mit Quoten für diese Heim-/Auswärts-Kombination gefunden.")

# -------------------------------------------------
# Eingaben: Quoten (manuell oder via CSV/Live)
# -------------------------------------------------
st.markdown(
    "### Wettquoten " +
    ("(**manuell eintragen**: für **zukünftige** Spiele)"
     if mode.startswith("Zukünftiges")
     else "(werden bei Bedarf **automatisch** aus CSV übernommen)")
)
st.caption("Niedrigere Quote = höhere implizite Wahrscheinlichkeit. Unten entfernen wir die Marge (Overround) und normieren auf 100 %.")

c1, c2, c3 = st.columns(3)
with c1:
    B365H = st.number_input("B365H (Heimsieg)",
        value=float(st.session_state.get("B365H", 1.80)), step=0.01, key="B365H")
with c2:
    B365D = st.number_input("B365D (Unentschieden)",
        value=float(st.session_state.get("B365D", 3.60)), step=0.01, key="B365D")
with c3:
    B365A = st.number_input("B365A (Auswärtssieg)",
        value=float(st.session_state.get("B365A", 4.20)), step=0.01, key="B365A")

c4, c5, c6 = st.columns(3)
with c4:
    PSH = st.number_input("PSH (Heimsieg)",
        value=float(st.session_state.get("PSH", 1.85)), step=0.01, key="PSH")
with c5:
    PSD = st.number_input("PSD (Unentschieden)",
        value=float(st.session_state.get("PSD", 3.50)), step=0.01, key="PSD")
with c6:
    PSA = st.number_input("PSA (Auswärtssieg)",
        value=float(st.session_state.get("PSA", 4.00)), step=0.01, key="PSA")

# -------------------------------------------------
# Debug: Eingabe prüfen
# -------------------------------------------------
with st.expander("🔎 Debug: Eingabe-Features ansehen"):
    st.write({
        "HomeTeam": home_team, "AwayTeam": away_team,
        "B365H": B365H, "B365D": B365D, "B365A": B365A,
        "PSH": PSH, "PSD": PSD, "PSA": PSA
    })

# -------------------------------------------------
# Helper-Funktionen
# -------------------------------------------------
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df["B365_HminusA"] = df["B365H"] - df["B365A"]
    df["PS_HminusA"]   = df["PSH"]  - df["PSA"]
    df["Draw_avg"]     = 0.5 * (df["B365D"].astype(float) + df["PSD"].astype(float))
    return df

def implied_probs_from_odds(h, d, a):
    pH, pD, pA = 1.0/h, 1.0/d, 1.0/a
    overround = pH + pD + pA
    return pH/overround, pD/overround, pA/overround, overround

# -------------------------------------------------
# Vorhersage
# -------------------------------------------------
go = st.button("Vorhersage starten", type="primary")

if go:
    sample = pd.DataFrame([{
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "B365H": B365H, "B365D": B365D, "B365A": B365A,
        "PSH" : PSH , "PSD" : PSD , "PSA" : PSA ,
    }])
    sample = add_derived_features(sample)

    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]
    classes = list(model.classes_)

    label_map = {"H": "Heimsieg (H)", "D": "Unentschieden (D)", "A": "Auswärtssieg (A)"}
    st.success(f"**Vorhersage:** {label_map.get(pred, pred)}")

    st.caption(
        "Die Balken sind **Modell-Wahrscheinlichkeiten** (ML). "
        "Die Tabelle unten sind **faire Buchmacher-Wahrscheinlichkeiten** (1/Quote, auf 100 % normiert) – "
        "rein mathematisch, **kein** ML-Output."
    )

    color_map = {"H": "#2ecc71", "D": "#f1c40f", "A": "#e74c3c"}
    df_plot = pd.DataFrame({
        "Klasse": [label_map.get(c, c) for c in classes],
        "Code": classes,
        "Wahrscheinlichkeit": proba
    })
    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X("Klasse:N", sort=["Heimsieg (H)","Unentschieden (D)","Auswärtssieg (A)"]),
            y=alt.Y("Wahrscheinlichkeit:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Code:N",
                            scale=alt.Scale(domain=["H","D","A"],
                                            range=[color_map["H"], color_map["D"], color_map["A"]]),
                            legend=None),
            tooltip=["Klasse", alt.Tooltip("Wahrscheinlichkeit:Q", format=".2%")]
        )
        .properties(width=600, height=280)
    )
    st.subheader("Wahrscheinlichkeiten:")
    st.altair_chart(chart, use_container_width=True)
    st.caption("Hinweis: Die Klassenreihenfolge entspricht der Reihenfolge im Training (`model.classes_`).")

    # Faire Wahrscheinlichkeiten – Quelle korrekt labeln
    has_b365 = all(k in st.session_state for k in ("B365H","B365D","B365A"))
    has_ps   = all(k in st.session_state for k in ("PSH","PSD","PSA"))
    if has_b365:
        h_q, d_q, a_q = st.session_state["B365H"], st.session_state["B365D"], st.session_state["B365A"]
        label = "Bet365"
    elif has_ps:
        h_q, d_q, a_q = st.session_state["PSH"], st.session_state["PSD"], st.session_state["PSA"]
        label = "Pinnacle (PS)"
    else:
        h_q = st.session_state.get("B365H", st.session_state.get("PSH", B365H))
        d_q = st.session_state.get("B365D", st.session_state.get("PSD", B365D))
        a_q = st.session_state.get("B365A", st.session_state.get("PSA", B365A))
        label = "Quelle (gemischt / manuell)"

    st.subheader(f"Quoten → (faire) Wahrscheinlichkeiten – {label}")
    pH, pD, pA, overround = implied_probs_from_odds(h_q, d_q, a_q)
    tbl = pd.DataFrame({
        "Ereignis": ["Heimsieg (H)", "Unentschieden (D)", "Auswärtssieg (A)"],
        f"Quote ({label})": [h_q, d_q, a_q],
        "Implied p (unnormiert)": [1/h_q, 1/d_q, 1/a_q],
        "Faire p (normiert)": [pH, pD, pA]
    })
    st.table(tbl.style.format({
        f"Quote ({label})": "{:.2f}",
        "Implied p (unnormiert)": "{:.3f}",
        "Faire p (normiert)": "{:.3f}"
    }))
    st.caption(f"Overround {label}: {overround:.3f}. "
               "Je größer >1, desto höher die Buchmachermarge. Wir normalisieren auf Summe=1.")
