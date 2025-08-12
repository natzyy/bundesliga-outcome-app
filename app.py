import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt

st.set_page_config(page_title="Bundesliga Spielergebnis", page_icon="⚽", layout="centered")
from pathlib import Path

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
    df = pd.concat(dfs, ignore_index=True)

    # Nur Zeilen mit kompletten Quoten behalten
    req = ["HomeTeam","AwayTeam","B365H","B365D","B365A","PSH","PSD","PSA"]
    df = df.dropna(subset=[c for c in req if c in df.columns])

    # Datum parsen (falls vorhanden)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df

# Pfade zu deinen Dateien im Repo (unter /data)
DATA_PATHS = [
    "data/D1_2020_2021.csv",
    "data/D1_2021_2022.csv",
    "data/D1_2022_2023.csv",
    "data/D1_2023_2024.csv",
    "data/D1_2024_2025.csv",
]

df_all = load_matches(DATA_PATHS)

# -------------------------
# Modell laden + Teamliste
# -------------------------
model = joblib.load("bundesliga_best_model.joblib")

# Versuche Teams aus dem OneHotEncoder zu holen
try:
    ohe = model.named_steps["preprocess"].named_transformers_["cat"]
    # Reihenfolge entspricht ["HomeTeam","AwayTeam"]
    home_teams = list(ohe.categories_[0])
    away_teams = list(ohe.categories_[1])
    teams = sorted(set(home_teams) | set(away_teams))
except Exception:
    # Fallback: freie Eingabe
    teams = None

st.title("⚽ Bundesliga Spielergebnis\nVorhersage")

# -------------------------
# Eingaben
# -------------------------
colA, colB = st.columns(2)
with colA:
    if teams:
        home_team = st.selectbox("Heimmannschaft", teams, index=teams.index("Bayern Munich") if "Bayern Munich" in teams else 0)
    else:
        home_team = st.text_input("Heimmannschaft", "Bayern Munich")

with colB:
    if teams:
        away_team = st.selectbox("Auswärtsmannschaft", teams, index=teams.index("Borussia Dortmund") if "Borussia Dortmund" in teams else 0)
    else:
        away_team = st.text_input("Auswärtsmannschaft", "Borussia Dortmund")
# --- Auto-Lookup historischer Quoten aus CSVs ---
cands = df_all[(df_all["HomeTeam"] == home_team) & (df_all["AwayTeam"] == away_team)].copy()
selected_row = None

if not cands.empty:
    # Falls mehrere Spiele: Datum auswählen
    if "Date" in cands.columns and cands["Date"].notna().any():
        cands = cands.sort_values("Date", ascending=False)
        date_choices = [d.strftime("%Y-%m-%d") if pd.notna(d) else "ohne Datum" for d in cands["Date"]]
        picked_date = st.selectbox("Historisches Spiel wählen", date_choices, key="picked_date")
        selected_row = cands.iloc[date_choices.index(picked_date)]
    else:
        selected_row = cands.iloc[0]

    # Button: Quoten in die Eingabefelder übernehmen
    if st.button("🔁 Quoten aus CSV übernehmen"):
        for k in ["B365H","B365D","B365A","PSH","PSD","PSA"]:
            if k in selected_row and not pd.isna(selected_row[k]):
                st.session_state[k] = float(selected_row[k])
        st.toast("Quoten aus CSV übernommen.", icon="✅")
else:
    st.info("Kein historisches Spiel mit Quoten für diese Heim-/Auswärts-Kombination gefunden.")


st.markdown("### Wettquoten (werden bei Bedarf aus CSVs geladen)")

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


# Button
go = st.button("Vorhersage starten", type="primary")

# Debug accordion (optional)
with st.expander("🔎 Debug: Eingabe-Features ansehen"):
    st.write({
        "HomeTeam": home_team, "AwayTeam": away_team,
        "B365H": B365H, "B365D": B365D, "B365A": B365A,
        "PSH": PSH, "PSD": PSD, "PSA": PSA
    })

# -------------------------
# Helper-Funktionen
# -------------------------
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

# -------------------------
# Vorhersage
# -------------------------
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
    classes = list(model.classes_)         # z.B. ["A","D","H"] – Reihenfolge beachten!

    # Lesbar ausgeben
    label_map = {"H": "Heimsieg (H)", "D": "Unentschieden (D)", "A": "Auswärtssieg (A)"}
    st.success(f"**Vorhersage:** {label_map.get(pred, pred)}")

    # Balken – kräftige Farben
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
                            scale=alt.Scale(domain=["H","D","A"], range=[color_map["H"], color_map["D"], color_map["A"]]),
                            legend=None),
            tooltip=["Klasse", alt.Tooltip("Wahrscheinlichkeit:Q", format=".2%")]
        )
        .properties(width=600, height=280)
    )
    st.subheader("Wahrscheinlichkeiten:")
    st.altair_chart(chart, use_container_width=True)
    st.caption("Hinweis: Die Klassenreihenfolge entspricht der Reihenfolge im Training (model.classes_).")

    # Quoten -> faire Wahrscheinlichkeiten
    st.subheader("Quoten → (faire) Wahrscheinlichkeiten")
    pH, pD, pA, overround = implied_probs_from_odds(B365H, B365D, B365A)
    tbl = pd.DataFrame({
        "Ereignis": ["Heimsieg (H)", "Unentschieden (D)", "Auswärtssieg (A)"],
        "Quote (B365)": [B365H, B365D, B365A],
        "Implied p (unnormiert)": [1/B365H, 1/B365D, 1/B365A],
        "Faire p (normiert)": [pH, pD, pA]
    })
    st.table(tbl.style.format({"Quote (B365)": "{:.2f}", "Implied p (unnormiert)": "{:.3f}", "Faire p (normiert)": "{:.3f}"}))
    st.caption(f"Overround (Summe der inversen Quoten) = {overround:.3f}. "
               "Je größer >1, desto höher die Buchmachermarge. Wir normalisieren auf Summe=1.")
