import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import statsmodels.api as sm

# Fonction pour convertir une durée (hh:mm:ss ou mm:ss) en secondes
def time_to_seconds(time_str):
    """Convertit une durée au format H:M:S ou M:S en secondes."""
    if isinstance(time_str, str):
        parts = list(map(int, time_str.split(":")))
        return sum(x * 60**i for i, x in enumerate(reversed(parts)))
    return time_str

# Charger les données en tenant compte du séparateur ";"
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/Bruns/Downloads/running_data.csv", encoding="utf-8", sep=";")
    df.columns = df.columns.str.strip()  # Supprime les espaces autour des noms de colonnes
    
    # Convertir la colonne "Date" au format datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

    # Convertir l'allure en secondes/km
    df["Allure (s/km)"] = df["Allure (min/km)"].apply(time_to_seconds)
    
    return df

data = load_data()



# 🔹 Gestion des données via session_state
if "run_data" not in st.session_state:
    st.session_state.run_data = data.copy()

data = st.session_state.run_data
data["Date_num"] = (data["Date"] - data["Date"].min()).dt.days  

# 🔹 Régression linéaire globale
X = sm.add_constant(data["Date_num"])
y = data["Allure (s/km)"]  # ✅ Correction ici
model = sm.OLS(y, X).fit()
data["Regression"] = model.predict(X)

# 🔹 Définition des ticks Y pour lisibilité
min_pace = data["Allure (s/km)"].min()
max_pace = data["Allure (s/km)"].max()
tick_range = np.arange(int(min_pace), int(max_pace) + 1, 10)  # Ticks toutes les 10s

# Fonction inverse pour afficher en min/km
def seconds_to_pace(seconds):
    minutes = seconds // 60
    sec = seconds % 60
    return f"{int(minutes)}:{int(sec):02d}"

tick_labels = [seconds_to_pace(t) for t in tick_range]

# 📊 **Graphique 1 : Évolution avec régression**
fig1 = px.scatter(data, x="Date", y="Allure (s/km)", trendline="ols",
                  title="📉 Évolution de l'allure moyenne",
                  labels={"Allure (s/km)": "Allure (mm:ss/km)"})

fig1.update_yaxes(tickvals=tick_range, ticktext=tick_labels)
fig1.update_xaxes(tickangle=-45)

st.plotly_chart(fig1)

# 📊 **Graphique 2 : Évolution avec taille = distance et couleur = catégorie**
fig2 = px.scatter(data, x="Date", y="Allure (s/km)", size="Distance (km)", color="Catégorie",
                  title="📉 Évolution de l'allure (taille = distance, couleur = catégorie)",
                  labels={"Allure (s/km)": "Allure (mm:ss/km)"})

# Ajout de la **ligne de régression unique**
fig2.add_scatter(x=data["Date"], y=data["Regression"], mode="lines", name="Tendance", line=dict(color="black"))

fig2.update_yaxes(tickvals=tick_range, ticktext=tick_labels)
fig2.update_xaxes(tickangle=-45)

st.plotly_chart(fig2)

# 📊 **Graphique 3 : Évolution avec taille = distance et couleur = fréquence cardiaque**
fig3 = px.scatter(data, x="Date", y="Allure (s/km)", size="Distance (km)", color="FC (bpm avg)", 
                  color_continuous_scale="RdYlGn_r",  # Rouge = FC haute, Vert = FC basse
                  title="📉 Évolution de l'allure (taille = distance, couleur = FC)",
                  labels={"Allure (s/km)": "Allure (mm:ss/km)"})

fig3.add_scatter(x=data["Date"], y=data["Regression"], mode="lines", name="Tendance", line=dict(color="black"))

fig3.update_yaxes(tickvals=tick_range, ticktext=tick_labels)
fig3.update_xaxes(tickangle=-45)

st.plotly_chart(fig3)

