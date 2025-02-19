import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cvxpy as cp
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI

# --- 1. Récupération des données de prix via CoinGecko ---

@st.cache_data(ttl=3600)
def fetch_crypto_data(crypto_list, days=365, vs_currency='usd'):
    """
    Récupère les données historiques de prix pour chaque crypto de la liste.
    Retourne un DataFrame indexé par la date, avec une colonne par crypto.
    """
    cg = CoinGeckoAPI()
    dfs = []
    for coin in crypto_list:
        try:
            data = cg.get_coin_market_chart_by_id(id=coin, vs_currency=vs_currency, days=days)
            prices = data['prices']  # liste de [timestamp, prix]
            df = pd.DataFrame(prices, columns=['timestamp', coin])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            dfs.append(df)
        except Exception as e:
            st.error(f"Erreur lors de la récupération de {coin}: {e}")
    if dfs:
        df_all = pd.concat(dfs, axis=1)
        df_all = df_all.sort_index()
        return df_all
    else:
        return pd.DataFrame()

# --- 2. Optimisation de portefeuille (Markowitz) ---

def markowitz_optimization(returns, risk_free_rate=0.0):
    """
    Calcule l’optimisation du portefeuille en maximisant le ratio de Sharpe.
    Contraintes : somme des poids = 1 et poids >= 0 (pas de short-selling).
    """
    mu = returns.mean().values
    sigma = returns.cov().values
    n = len(mu)
    
    # Définition de la variable d'optimisation (poids)
    w = cp.Variable(n)
    port_return = mu @ w
    port_vol = cp.quad_form(w, sigma) ** 0.5
    sharpe = (port_return - risk_free_rate) / port_vol
    
    problem = cp.Problem(cp.Maximize(sharpe),
                         [cp.sum(w) == 1,
                          w >= 0])
    problem.solve()
    
    optimal_weights = w.value
    expected_sharpe = sharpe.value
    return dict(zip(returns.columns, optimal_weights)), expected_sharpe

# --- 3. Heatmap de corrélation ---

def plot_correlation_heatmap(data):
    corr = data.pct_change().corr()
    fig = px.imshow(corr,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title="Matrice de Corrélation des Rendements")
    return fig

# --- 4. Simulation "What-If" ---

def simulate_scenarios(data, shock=-0.2, days=30):
    """
    Simule un scénario de marché : application d’un choc (en pourcentage)
    sur les prix du dernier jour, suivi d’une reprise linéaire sur 'days' jours.
    Retourne un DataFrame simulé pour chaque crypto.
    """
    last_prices = data.iloc[-1]
    shocked_prices = last_prices * (1 + shock)
    
    sim_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=days)
    sim_data = pd.DataFrame(index=sim_dates, columns=data.columns)
    
    for asset in data.columns:
        sim_data[asset] = np.linspace(shocked_prices[asset], last_prices[asset], days)
    
    return sim_data

# --- Interface Streamlit ---

st.title("Portfolio Management Crypto")

# --- 1. Gestion dynamique du portfolio et récupération des données ---

st.sidebar.header("Configuration du Portfolio")

# Saisie de la liste des cryptos (les identifiants CoinGecko en minuscules)
crypto_input = st.sidebar.text_input("Liste des cryptos (CoinGecko IDs, séparés par des virgules)", 
                                     "bitcoin, ethereum, litecoin")
crypto_list = [coin.strip().lower() for coin in crypto_input.split(",") if coin.strip() != ""]

# Choix du nombre de jours historiques à récupérer
days = st.sidebar.number_input("Nombre de jours historiques", min_value=30, max_value=1095, value=365, step=1)

# Récupération des données de prix
with st.spinner("Récupération des données depuis CoinGecko ..."):
    price_data = fetch_crypto_data(crypto_list, days=days, vs_currency='usd')

if price_data.empty:
    st.error("Aucune donnée n'a pu être récupérée. Vérifie les identifiants des cryptos.")
    st.stop()

st.header("1. Évolution des Prix Historiques")
st.line_chart(price_data)

# Calcul du rendement du portefeuille si on applique les poids optimaux
# --- 2. Optimisation de Portefeuille (Markowitz) ---

st.header("2. Optimisation de Portefeuille (Markowitz)")
returns_df = price_data.pct_change().dropna()
optimal_weights, exp_sharpe = markowitz_optimization(returns_df)

st.write("Poids optimaux recommandés :")
st.write(optimal_weights)
st.write(f"Ratio de Sharpe attendu : {exp_sharpe:.2f}")

# Calcul de la valeur du portefeuille sur l'historique
# On suppose une valeur initiale (par exemple 1.0) multipliée par les poids
portfolio_value = price_data.copy()
for asset in portfolio_value.columns:
    portfolio_value[asset] = portfolio_value[asset] * optimal_weights.get(asset, 0)
portfolio_value["Total"] = portfolio_value.sum(axis=1)

st.subheader("Évolution de la Valeur du Portefeuille")
st.line_chart(portfolio_value["Total"])

# --- 3. Analyse de Corrélation et Diversification ---

st.header("3. Analyse de Corrélation et Diversification")
fig_corr = plot_correlation_heatmap(price_data)
st.plotly_chart(fig_corr, use_container_width=True)

# --- 4. Simulation "What-If" et Scénarios de Marché ---

st.header("4. Simulations 'What-If' et Scénarios de Marché")
shock_val = st.slider("Intensité du choc (%)", min_value=-50, max_value=0, value=-30, step=1)
simulated_data = simulate_scenarios(price_data, shock=shock_val/100, days=30)
st.subheader("Évolution simulée des prix (après choc)")
st.line_chart(simulated_data)

# Impact sur la valeur du portefeuille optimisé
sim_portfolio_value = simulated_data.copy()
for asset in simulated_data.columns:
    sim_portfolio_value[asset] = simulated_data[asset] * optimal_weights.get(asset, 0)
sim_portfolio_value["Total"] = sim_portfolio_value.sum(axis=1)

st.subheader("Impact sur la Valeur du Portefeuille Optimisé")
last_real_value = portfolio_value["Total"].iloc[-1]
sim_final_value = sim_portfolio_value["Total"].iloc[-1]
variation = ((sim_final_value - last_real_value) / last_real_value) * 100

st.write(f"Valeur du portefeuille avant le choc : {last_real_value:.2f} $")
st.write(f"Valeur du portefeuille après simulation : {sim_final_value:.2f} $")
st.write(f"Variation : {variation:.2f} %")
st.line_chart(sim_portfolio_value["Total"])
