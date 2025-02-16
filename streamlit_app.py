import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
import statsmodels.api as sm
from datetime import datetime

def pace_to_seconds(pace):
    """Convert pace from mm:ss to total seconds"""
    minutes, seconds = map(int, pace.split(':'))
    return minutes * 60 + seconds

def seconds_to_pace(seconds):
    """Convert total seconds to mm:ss pace format"""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{int(minutes)}:{int(seconds):02d}"

data = pd.read_csv("running_data.csv")  # Charger les données depuis un fichier CSV

data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data['Pace (s/km)'] = data['Pace (min/km)'].apply(pace_to_seconds)

def plot_regression(ax, x, y, color='black'):
    """Ajoute une régression linéaire au graphique"""
    x_num = mdates.date2num(x)
    X = sm.add_constant(x_num)
    model = sm.OLS(y, X).fit()
    ax.plot(x, model.predict(X), color=color, linestyle='dashed', linewidth=2)

plt.figure(figsize=(18, 12))

# 1 - Scatter plot avec régression globale
ax1 = plt.subplot(3, 1, 1)
sns.scatterplot(x='Date', y='Pace (s/km)', data=data, ax=ax1, color='blue')
plot_regression(ax1, data['Date'], data['Pace (s/km)'])
ax1.set_title("Évolution de l'allure moyenne dans le temps")
ax1.set_ylabel("Allure (min/km)")
ax1.set_xlabel("Date")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
ax1.set_yticks(range(int(min(data['Pace (s/km)'])), int(max(data['Pace (s/km)'])), 10))
ax1.set_yticklabels([seconds_to_pace(t) for t in ax1.get_yticks()])
plt.xticks(rotation=45)

# 2 - Scatter plot avec taille = distance, couleur = Length, régression globale
ax2 = plt.subplot(3, 1, 2)
sns.scatterplot(x='Date', y='Pace (s/km)', size='Distance (km)', hue='Length', data=data, ax=ax2, sizes=(20, 200))
plot_regression(ax2, data['Date'], data['Pace (s/km)'])
ax2.set_title("Évolution de l'allure avec distance et type de course")
ax2.set_ylabel("Allure (min/km)")
ax2.set_xlabel("Date")
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
ax2.set_yticks(range(int(min(data['Pace (s/km)'])), int(max(data['Pace (s/km)'])), 10))
ax2.set_yticklabels([seconds_to_pace(t) for t in ax2.get_yticks()])
plt.xticks(rotation=45)

# 3 - Scatter plot avec taille = distance, couleur = fréquence cardiaque moyenne
ax3 = plt.subplot(3, 1, 3)
sns.scatterplot(x='Date', y='Pace (s/km)', size='Distance (km)', hue='FC (bpm avg)', data=data, ax=ax3, sizes=(20, 200), palette='RdYlGn_r')
plot_regression(ax3, data['Date'], data['Pace (s/km)'])
ax3.set_title("Évolution de l'allure avec fréquence cardiaque")
ax3.set_ylabel("Allure (min/km)")
ax3.set_xlabel("Date")
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
ax3.set_yticks(range(int(min(data['Pace (s/km)'])), int(max(data['Pace (s/km)'])), 10))
ax3.set_yticklabels([seconds_to_pace(t) for t in ax3.get_yticks()])
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
