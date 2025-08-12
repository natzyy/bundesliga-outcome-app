import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt
import os, requests
from pathlib import Path

# -------------------------------------------------
# Seite konfigurieren
# -------------------------------------------------
st.set_page_config(page_title="Bundesliga Spielergebnis", page_icon="⚽", layout="centered")

# --- DEBUG: zeigt, ob die CSVs da sind ---
import os, glob
VERSION = "UI-Mode + CSV debug"
st.sidebar.success(f"Build: {VERSION}")
st.sidebar.write("CWD:", os.getcwd())
st.sidebar.write("data/*.csv:", glob.glob("data/*.csv"))

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

    # nur Zeilen mit kompletten Quoten behalten
    req = ["HomeTeam","AwayTeam","B365H","B365D","B365A","PSH","PSD","PSA"]
    df_all = df_all.dropna(subset=[c for c in req if c in df_all.columns])

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
# Live-Quoten (The Odds API) holen
# -------------------------------------------------
@st.cache_data(ttl=300)
def fetch_live_odds(home_team, away_team, sport_key="soccer_germany_bundesliga", want_books=("pinnacle","bet365")):
    import re, unicodedata, os, requests, streamlit as st

    api_key = st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", "")).strip()
    if not api_key or "DEIN_" in api_key.upper():
        raise RuntimeError("ODDS_API_KEY fehlt/Platzhalter. In Streamlit > Settings > Secrets eintragen.")

    def norm(s: str) -> str:
        s = s or ""
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = s.lower()
        s = re.sub(r"[^a-z0-9 ]", " ", s)
        s = re.sub(r"\b(fc|sv|tsg|vfl|sc|borussia|bayer|1|04|05|1899|98|04)\b", " ", s)
        s = " ".join(s.split())
        return s

    # Aliase (kannst du jederzeit erweitern)
    ALIASES = {
        "bayern munich": ["bayern munich", "bayern munchen", "fc bayern munchen", "bayern"],
        "augsburg": ["augsburg", "fc augsburg"],
        "borussia dortmund": ["borussia dortmund", "dortmund"],
        "rb leipzig": ["rb leipzig", "leipzig", "rasenballsport leipzig"],
        "bayer leverkusen": ["bayer leverkusen", "leverkusen"],
        "borussia monchengladbach": ["borussia monchengladbach", "monchengladbach", "gladbach"],
        "koln": ["koln", "fc koln", "1 fc koln", "koln 1 fc", "kolner"],
        "union berlin": ["union berlin", "1 fc union berlin", "union"],
        "eintracht frankfurt": ["eintracht frankfurt", "frankfurt"],
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

    def alias_set(s):
        k = norm(s)
        return set([k] + ALIASES.get(k, []))

    nh, na = alias_set(home_team), alias_set(away_team)

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key, "regions": "eu,uk", "oddsFormat": "decimal",
        "markets": "h2h", "bookmakers": ",".join(want_books)
    }
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()  # v4: Liste mit home_team / away_team

    match = None
    flip = False  # ob Event-Home != User-Home (für 'Home'/'Away'-Mapping)
    for ev in data:
        ev_home = norm(ev.get("home_team", ""))
        ev_away = norm(ev.get("away_team", ""))

        # Treffer, wenn beide User-Teams im Event vorkommen (egal in welcher Richtung)
        if (ev_home in nh and ev_away in na) or (ev_home in na and ev_away in nh):
            match = ev
            # Wenn Event-Home das User-Away ist, müssen wir 'Home'/'Away' später flippen
            flip = (ev_home in na and ev_away in nh)
            break

    if not match:
        raise LookupError("Kein passendes Event für diese Teams gefunden (evtl. kein anstehendes Spiel oder Namensvariante).")

    res = {}
    for bk in match.get("bookmakers", []):
        bname = (bk.get("key","") or "").lower()
        if bname not in want_books:
            continue
        for m in bk.get("markets", []):
            if m.get("key") != "h2h":
                continue
            for out in m.get("outcomes", []):
                nm_raw = out.get("name","")
                nm = norm(nm_raw)
                price = float(out.get("price"))
                code = None

                if nm in ("draw", "unentschieden"):
                    code = "D"
                elif nm in nh:
                    code = "H"
                elif nm in na:
                    code = "A"
                elif nm in ("home","heim"):
                    code = "A" if flip else "H"
                elif nm in ("away","auswaerts","auswarts","auswarts","auswarts","gast","away team"):
                    code = "H" if flip else "A"

                if not code:
                    continue

                if "

# -------------------------------------------------
# Modell laden + Teamliste aus OneHotEncoder extrahieren
# -------------------------------------------------
model = joblib.load("bundesliga_best_model.joblib")

try:
    ohe = model.named_steps["preprocess"].named_transformers_["cat"]
    # Reihenfolge entspricht ["HomeTeam","AwayTeam"]
    home_teams = list(ohe.categories_[0])
    away_teams = list(ohe.categories_[1])
    teams = sorted(set(home_teams) | set(away_teams))
except Exception:
    teams = None  # Fallback auf Freitext

# -------------------------------------------------
# UI: Titel + Moduswahl
# -------------------------------------------------
st.title("⚽ Bundesliga Spielergebnis – Vorhersage")

has_history = not df_all.empty
choices = ["Zukünftiges Spiel (manuelle Quoten)"] + (["Historisches Spiel (Quoten aus CSV)"] if has_history else [])
mode = st.radio(
    "Vorhersage-Modus",
    choices,
    index=0,
    help="Historischer Modus holt nur bequem die damaligen Quoten. Die Vorhersage basiert IMMER auf einem Modell, das auf vielen vergangenen Jahren trainiert wurde."
)


# Trainingszeitraum aus CSVs anzeigen (nur als Info)
train_span = ""
if "Date" in df_all.columns and df_all["Date"].notna().any():
    dmin = df_all["Date"].min()
    dmax = df_all["Date"].max()
    if pd.notna(dmin) and pd.notna(dmax):
        train_span = f"Trainiert auf historischen Spielen ca. {dmin.date()} bis {dmax.date()}."

st.info(
    "ℹ️ **Wichtig:** Die Vorhersage kommt aus einem **trainierten ML-Modell** "
    "(Logistic Regression oder Random Forest). "
    "Der **historische Modus** dient nur dazu, die damaligen **Quoten automatisch** zu übernehmen. "
    + (f"\n\n{train_span}" if train_span else "")
)

# -------------------------
# Eingaben (Teams)
# -------------------------
colA, colB = st.columns(2)

with colA:
    if teams:
        home_team = st.selectbox(
            "Heimmannschaft",
            teams,
            index=teams.index("Bayern Munich") if "Bayern Munich" in teams else 0,
            key="home_team_select"
        )
    else:
        home_team = st.text_input("Heimmannschaft", "Bayern Munich")

with colB:
    if teams:
        away_team = st.selectbox(
            "Auswärtsmannschaft",
            teams,
            index=teams.index("Borussia Dortmund") if "Borussia Dortmund" in teams else 0,
            key="away_team_select"
        )
    else:
        away_team = st.text_input("Auswärtsmannschaft", "Borussia Dortmund")

# -------------------------
# Live-Quoten (nur im Zukunfts-Modus)
# -------------------------
# --- Live-Quoten (nur im Zukunfts-Modus) ---
if mode.startswith("Zukünftiges"):
    # Button: Live-Quoten laden
    if st.button("🔄 Live-Quoten laden (Bet365 & Pinnacle)"):
        try:
            odds, missing = fetch_live_odds(home_team, away_team)
            for k, v in odds.items():
                st.session_state[k] = float(v)
            if missing:
                st.warning("Nicht alle Quoten verfügbar: " + ", ".join(missing))
            else:
                st.success("Live-Quoten geladen.")
        except Exception as e:
            st.error(f"Live-Quoten konnten nicht geladen werden: {e}")

    # Optional: aktuell verfügbare Spiele anzeigen
 with st.expander("Welche Spiele kennt die API gerade?"):
    try:
        api_key = st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", "")).strip()
        if api_key:
            url = "https://api.the-odds-api.com/v4/sports/soccer_germany_bundesliga/odds"
            params = {"apiKey": api_key, "regions":"eu,uk", "oddsFormat":"decimal", "markets":"h2h"}
            resp = requests.get(url, params=params, timeout=12)
            resp.raise_for_status()
            items = resp.json()
            rows = []
            for it in items:
                h = it.get("home_team") or (it.get("teams",[None,None])[0] if it.get("teams") else None)
                a = it.get("away_team") or (it.get("teams",[None,None])[1] if it.get("teams") else None)
                rows.append(f"{h} vs {a} — {it.get('commence_time','')}")
            st.write(rows or "Keine kommenden Bundesliga-Spiele gefunden.")
        else:
            st.info("Kein ODDS_API_KEY gesetzt.")
    except Exception as e:
        st.warning(f"Listing fehlgeschlagen: {e}")


# -------------------------
# Historischer CSV-Lookup (nur im Historik-Modus)
# -------------------------
if mode == "Historisches Spiel (Quoten aus CSV)":
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
# Eingaben: Quoten (manuell oder via CSV-Autofill)
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
    # exakt wie im Notebook erzeugt:
    df["B365_HminusA"] = df["B365H"] - df["B365A"]
    df["PS_HminusA"]   = df["PSH"]  - df["PSA"]
    df["Draw_avg"]     = 0.5 * (df["B365D"].astype(float) + df["PSD"].astype(float))
    return df

def implied_probs_from_odds(h, d, a):
    # rohe inverse Quoten
    pH, pD, pA = 1.0/h, 1.0/d, 1.0/a
    overround = pH + pD + pA
    # Normalisieren (Buchmacher-Marge entfernen)
    return pH/overround, pD/overround, pA/overround, overround

# -------------------------------------------------
# Vorhersage
# -------------------------------------------------
go = st.button("Vorhersage starten", type="primary")

if go:
    # Input-DF
    sample = pd.DataFrame([{
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "B365H": B365H, "B365D": B365D, "B365A": B365A,
        "PSH" : PSH , "PSD" : PSD , "PSA" : PSA ,
    }])
    sample = add_derived_features(sample)

    # Modell-Prediction
    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]
    classes = list(model.classes_)  # z. B. ["A","D","H"] – Reihenfolge beachten!

    # Ausgabe
    label_map = {"H": "Heimsieg (H)", "D": "Unentschieden (D)", "A": "Auswärtssieg (A)"}
    st.success(f"**Vorhersage:** {label_map.get(pred, pred)}")

    # Kontext zur Vorhersage
    st.caption(
        "Die Balken sind **Modell-Wahrscheinlichkeiten** aus dem trainierten ML-Modell "
        "(Muster aus vielen vergangenen Spielen: Teams als One-Hot + Quoten + abgeleitete Features). "
        "Die Tabelle unten zeigt **faire Buchmacher-Wahrscheinlichkeiten** (1/Quote, auf 100 % normiert) – "
        "rein mathematisch, **kein** ML-Output."
    )

    # Balkendiagramm
    color_map = {"H": "#2ecc71", "D": "#f1c40f", "A": "#e74c3c"}  # grün, gelb, rot
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

    # Quoten -> faire Wahrscheinlichkeiten (Beispielhaft mit B365)
    st.subheader("Quoten → (faire) Wahrscheinlichkeiten")
    pH, pD, pA, overround = implied_probs_from_odds(B365H, B365D, B365A)
    tbl = pd.DataFrame({
        "Ereignis": ["Heimsieg (H)", "Unentschieden (D)", "Auswärtssieg (A)"],
        "Quote (B365)": [B365H, B365D, B365A],
        "Implied p (unnormiert)": [1/B365H, 1/B365D, 1/B365A],
        "Faire p (normiert)": [pH, pD, pA]
    })
    st.table(tbl.style.format({
        "Quote (B365)": "{:.2f}",
        "Implied p (unnormiert)": "{:.3f}",
        "Faire p (normiert)": "{:.3f}"
    }))
    st.caption(f"Overround (Summe der inversen Quoten) = {overround:.3f}. "
               "Je größer >1, desto höher die Buchmachermarge. Wir normalisieren auf Summe=1.")

    # Edu-Expander
    with st.expander("🧠 Wie funktioniert die Vorhersage?"):
        st.markdown("""
- **Training:** auf vielen historischen Spielen (Teams → One-Hot, Quoten, abgeleitete Merkmale).
- **Vorhersage:** nutzt **nur** die aktuell eingegebenen Teams & Quoten, nicht das Datum.
- **Historischer Modus:** holt **nur** damals gültige Quoten automatisch.
- **Vergleich:** „faire“ Quoten-Prozente sind 1/Quote, auf 100 % normiert (Overround entfernt).
""")

