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
    df = pd.read_csv("running_data.csv", encoding="utf-8", sep=";")
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

# 🔹 Sidebar Form for Adding a New Run
st.sidebar.title("🏃 Ajouter une nouvelle course")
with st.sidebar.form("add_run_form"):
    new_date = st.date_input("📅 Date", value=pd.to_datetime("today"))
    new_duration = st.text_input("⏳ Durée (hh:mm:ss ou mm:ss)", "00:30:00")
    new_distance = st.number_input("📏 Distance (km)", min_value=0.1, step=0.1)
    new_pace = st.text_input("⏱️ Allure (mm:ss/km)", "05:30")
    new_best_km = st.text_input("🏅 Best Km (mm:ss)", "04:50")
    new_fc = st.number_input("❤️ FC (bpm avg)", min_value=40, max_value=220, step=1)
    new_category = st.selectbox("🏷️ Catégorie", ["Easy", "Tempo", "Interval", "Long Run"])
    
    submitted = st.form_submit_button("➕ Ajouter")

    if submitted:
        # Convert time fields to seconds
        new_entry = pd.DataFrame({
            "Date": [new_date.strftime("%d/%m/%Y")],  # Keep as string to match CSV format
            "Durée": [new_duration],
            "Distance (km)": [new_distance],
            "Allure (min/km)": [new_pace],
            "Best Km": [new_best_km],
            "FC (bpm avg)": [new_fc],
            "Catégorie": [new_category]
        })
        
        # Append new data to CSV
        new_entry.to_csv(CSV_FILE, mode="a", header=not os.path.exists(CSV_FILE), index=False, sep=";")
        
        # Update session state and reload data
        st.session_state.run_data = pd.concat([st.session_state.run_data, new_entry], ignore_index=True)
        st.success("✅ Nouvelle course ajoutée et sauvegardée dans le fichier CSV !")

# Reload the updated data
data = st.session_state.run_data
data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")
data["Allure (s/km)"] = data["Allure (min/km)"].apply(time_to_seconds)
data["Date_num"] = (data["Date"] - data["Date"].min()).dt.days  


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

