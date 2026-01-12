import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt

# Charger modèle et scaler
model = joblib.load("modele_reussite.pkl")
scaler = joblib.load("scaler.pkl")

# Titre de l'application
st.title("🎓 IA – Prédiction de Réussite Scolaire")
st.write("Entrez les informations de l'élève pour prédire ses chances de réussite :")

# Inputs utilisateur
heures = st.number_input("Heures d'étude par semaine", 0.0, 20.0, 5.0, step=0.5)
absences = st.number_input("Nombre d'absences", 0, 50, 2)
devoirs = st.slider("Devoirs terminés (0–10)", 0, 10, 5)

# Bouton de prédiction
if st.button("Prédire"):
    # Préparer les données
    X = scaler.transform([[heures, absences, devoirs]])
    proba = model.predict_proba(X)[0][1] * 100
    proba_echec = 100 - proba

    # Affichage du résultat
    if proba >= 50:
        st.success(f"✅ Probabilité de réussite : {proba:.1f}%")
    else:
        st.error(f"❌ Risque d'échec : {proba_echec:.1f}%")

    # Visualisation avec Altair pour couleurs différentes
    df = pd.DataFrame({
        'Résultat': ['Réussite', 'Échec'],
        'Probabilité': [proba, proba_echec]
    })

    chart = alt.Chart(df).mark_bar().encode(
        x='Résultat',
        y='Probabilité',
        color=alt.condition(
            alt.datum.Résultat == 'Réussite',
            alt.value('green'),
            alt.value('red')
        )
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Barre verte = réussite, Barre rouge = échec")
