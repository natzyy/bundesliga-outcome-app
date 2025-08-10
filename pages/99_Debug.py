import os, traceback
import streamlit as st
import joblib
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="🔧 Debug", layout="wide")
st.title("🔧 Debug – Bundesliga Outcome App")

# --- 1) Umfeld prüfen
st.subheader("1) Umgebung")
st.write("📂 Arbeitsverzeichnis:", os.getcwd())
st.write("📄 Dateien im Verzeichnis:", os.listdir())

# --- 2) Modell laden (sichtbare Fehlerausgabe)
@st.cache_resource
def load_model():
    path = Path(__file__).resolve().parent.parent / "bundesliga_best_model.joblib"
    # Hinweis: __file__/parent.parent -> zurück aus pages/ ins Repo-Root
    if not path.exists():
        st.error(f"❌ Modell-Datei nicht gefunden: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error("❌ Fehler beim Laden des Modells:")
        st.code(traceback.format_exc())
        return None

model = load_model()
if model is not None:
    st.success("✅ Modell geladen.")
else:
    st.stop()

# --- 3) Feste Beispielvorhersage (Bayern vs. Dortmund)
st.subheader("2) Schnelltest – Beispiel Bayern vs. Dortmund")
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if all(c in df.columns for c in ["B365H","B365A"]):
        df["B365_HminusA"] = df["B365H"] - df["B365A"]
    if all(c in df.columns for c in ["PSH","PSA"]):
        df["PS_HminusA"]   = df["PSH"] - df["PSA"]
    if all(c in df.columns for c in ["B365D","PSD"]):
        df["Draw_avg"]     = 0.5*(df["B365D"].astype(float) + df["PSD"].astype(float))
    return df

example = pd.DataFrame([{
    "HomeTeam": "Bayern Munich", "AwayTeam": "Borussia Dortmund",
    "B365H": 1.80, "B365D": 3.60, "B365A": 4.20,
    "PSH":  1.85, "PSD":  3.50, "PSA":  4.00
}])
example = add_derived_features(example)
pred_ex = model.predict(example)[0]
proba_ex = model.predict_proba(example)[0]
classes  = list(model.classes_)

st.write(f"Vorhersage: **{pred_ex}**  (H=Heim, D=Remis, A=Auswärts)")
df_plot = pd.DataFrame({"Klasse": classes, "Wahrscheinlichkeit": proba_ex})
st.bar_chart(df_plot.set_index("Klasse"))

# --- 4) Manuelle Tests
st.subheader("3) Eigener Test – Eingaben")
colA, colB = st.columns(2)
home = colA.text_input("Heimmannschaft", "Bayern Munich")
away = colB.text_input("Auswärtsmannschaft", "Borussia Dortmund")

c1,c2,c3 = st.columns(3)
B365H = c1.number_input("B365H", value=1.80, min_value=1.01, step=0.05)
B365D = c2.number_input("B365D", value=3.60, min_value=1.01, step=0.05)
B365A = c3.number_input("B365A", value=4.20, min_value=1.01, step=0.05)

c4,c5,c6 = st.columns(3)
PSH = c4.number_input("PSH", value=1.85, min_value=1.01, step=0.05)
PSD = c5.number_input("PSD", value=3.50, min_value=1.01, step=0.05)
PSA = c6.number_input("PSA", value=4.00, min_value=1.01, step=0.05)

if st.button("➡️ Test ausführen"):
    sample = pd.DataFrame([{
        "HomeTeam": home, "AwayTeam": away,
        "B365H": B365H, "B365D": B365D, "B365A": B365A,
        "PSH": PSH, "PSD": PSD, "PSA": PSA
    }])
    sample = add_derived_features(sample)
    try:
        pred = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0]
        st.success(f"Vorhersage: **{pred}**")
        plot_df = pd.DataFrame({"Klasse": model.classes_, "Wahrscheinlichkeit": proba})
        st.bar_chart(plot_df.set_index("Klasse"))
    except Exception as e:
        st.error("Fehler bei der Vorhersage:")
        st.code(traceback.format_exc())
