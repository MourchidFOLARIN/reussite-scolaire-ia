import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Charger les données
df = pd.read_csv("dataset.csv")

# Séparer les features et le résultat
X = df[['study_hours', 'absences', 'assignments_done']]
y = df['final_result']

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Découpage en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Modèle Random Forest
model = RandomForestClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde du modèle et du scaler
joblib.dump(model, "modele_reussite.pkl")
joblib.dump(scaler, "scaler.pkl")

# Affichage de la précision du modèle
score = model.score(X_test, y_test)
print(f"✅ Modèle entraîné et sauvegardé avec précision : {score*100:.2f}%")
