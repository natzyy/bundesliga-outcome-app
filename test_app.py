import joblib
import pandas as pd

# Modell laden
model = joblib.load("bundesliga_best_model.joblib")

# Beispiel-Eingaben (Bayern vs. Dortmund)
home_team = "Bayern Munich"
away_team = "Borussia Dortmund"
B365H, B365D, B365A = 1.80, 3.60, 4.20
PSH, PSD, PSA = 1.85, 3.50, 4.00

# Zusatzfeatures berechnen
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df["B365_HminusA"] = df["B365H"] - df["B365A"]
    df["PS_HminusA"]   = df["PSH"]  - df["PSA"]
    df["Draw_avg"]     = 0.5 * (df["B365D"].astype(float) + df["PSD"].astype(float))
    return df

# Eingabe-Datenframe
sample = pd.DataFrame([{
    "HomeTeam": home_team,
    "AwayTeam": away_team,
    "B365H": B365H, "B365D": B365D, "B365A": B365A,
    "PSH": PSH, "PSD": PSD, "PSA": PSA
}])
sample = add_derived_features(sample)

# Vorhersage + Wahrscheinlichkeiten
pred = model.predict(sample)[0]
proba = model.predict_proba(sample)[0]
classes = list(model.classes_)

print("🔍 Testvorhersage")
print("----------------")
print(f"Spiel: {home_team} vs {away_team}")
print(f"Vorhersage: {pred}")
print("Wahrscheinlichkeiten:")
for cls, p in zip(classes, proba):
    print(f"  {cls}: {p:.2%}")
