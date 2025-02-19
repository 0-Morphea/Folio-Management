import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI

# -------------------------------
# Fonctions d'accès aux données
# -------------------------------

@st.cache_data(ttl=3600)
def get_top_coins(n=200, vs_currency='usd'):
    """
    Récupère les top n cryptos par capitalisation via CoinGecko.
    Retourne une DataFrame avec les colonnes: id, symbol, name.
    """
    cg = CoinGeckoAPI()
    coins = cg.get_coins_markets(vs_currency=vs_currency, order='market_cap_desc', per_page=n, page=1)
    df = pd.DataFrame(coins)
    return df[['id', 'symbol', 'name']]

@st.cache_data(ttl=3600)
def fetch_crypto_history(coin_id, days=365, vs_currency='usd'):
    """
    Récupère l'historique des prix pour une crypto sur les 12 derniers mois.
    Retourne un DataFrame indexé par la date, avec une colonne "price".
    """
    cg = CoinGeckoAPI()
    try:
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
        prices = data['prices']  # liste de [timestamp, price]
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"Erreur pour {coin_id}: {e}")
        return pd.DataFrame()

def get_current_price(coin_id, vs_currency='usd'):
    """
    Récupère le prix actuel d'une crypto.
    """
    cg = CoinGeckoAPI()
    try:
        data = cg.get_price(ids=coin_id, vs_currencies=vs_currency)
        return data[coin_id][vs_currency]
    except Exception as e:
        st.error(f"Erreur lors de la récupération du prix pour {coin_id}: {e}")
        return None

# -------------------------------
# Fonctions de simulation & forecast
# -------------------------------

def forecast_price_series(price_series, forecast_days=365, n_simulations=500):
    """
    Effectue une simulation Monte-Carlo pour forecast le cours futur d'une crypto sur 1 an.
    Utilise un mouvement Brownien géométrique.
    Retourne un DataFrame des simulations et les quantiles (5e, 50e, 95e).
    """
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()
    last_price = price_series.iloc[-1]
    dt = 1  # daily step

    sims = np.zeros((n_simulations, forecast_days+1))
    sims[:, 0] = last_price

    for t in range(1, forecast_days+1):
        shocks = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_simulations)
        sims[:, t] = sims[:, t-1] * np.exp(shocks)

    sim_df = pd.DataFrame(sims.T)
    sim_df.index = [price_series.index[-1] + timedelta(days=i) for i in range(forecast_days+1)]
    quantiles = sim_df.quantile([0.05, 0.50, 0.95]).T
    quantiles.columns = ['q05', 'q50', 'q95']
    return sim_df, quantiles

# -------------------------------
# Fonctions d'analyse de risque
# -------------------------------

def risk_metrics(price_series):
    """
    Calcule la volatilité annualisée et le max drawdown pour une série de prix.
    """
    returns = price_series.pct_change().dropna()
    volatility = returns.std() * np.sqrt(365)
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    return volatility, max_drawdown

# -------------------------------
# Dashboard Agrégé
# -------------------------------

def portfolio_allocation_pie(allocation_dict, cash_value):
    """
    Crée un pie chart de la répartition de l'allocation en capital.
    """
    labels = list(allocation_dict.keys()) + ['Cash']
    values = list(allocation_dict.values()) + [cash_value]
    fig = px.pie(names=labels, values=values, title="Répartition de l'Allocation du Capital")
    return fig

# -------------------------------
# Interface Streamlit
# -------------------------------

st.title("Portfolio Management & Forecasting pour Cryptos")

st.sidebar.header("Configuration du Portfolio")

# 1. Sélection parmi le Top 200
top_coins_df = get_top_coins(n=200)
coin_options = top_coins_df['name'] + " (" + top_coins_df['symbol'].str.upper() + ")"
coin_ids = top_coins_df['id'].tolist()
coin_dict = dict(zip(coin_options, coin_ids))

selected_coins = st.sidebar.multiselect("Sélectionnez les cryptos à inclure", options=coin_options,
                                          default=coin_options[:3])
selected_ids = [coin_dict[name] for name in selected_coins]

# 2. Montant total du portefeuille
total_portfolio = st.sidebar.number_input("Montant total du portefeuille ($)", min_value=1000.0, value=100000.0, step=1000.0)

# 3. Allocation entre cash et cryptos
cash_allocation = st.sidebar.slider("Pourcentage d'allocation en cash", min_value=0, max_value=100, value=20, step=5)
crypto_allocation_pct = 100 - cash_allocation

st.sidebar.markdown(f"Allocation globale en cryptos : **{crypto_allocation_pct}%**")

# 4. Pour chaque crypto sélectionnée, définir l'allocation spécifique en %
allocation_inputs = {}
if selected_ids:
    st.sidebar.markdown("### Allocation par Crypto")
    total_alloc = 0
    for coin_name in selected_coins:
        alloc = st.sidebar.number_input(f"Allocation pour {coin_name} (%)", min_value=0.0, max_value=100.0, value=round(crypto_allocation_pct/len(selected_coins),1), step=0.5, key=coin_name)
        allocation_inputs[coin_name] = alloc
        total_alloc += alloc
    # Normaliser si nécessaire
    if total_alloc != 0:
        for coin in allocation_inputs:
            allocation_inputs[coin] = allocation_inputs[coin] / total_alloc * crypto_allocation_pct

# 5. Yield sur le cash
yield_cash = st.sidebar.number_input("Yield annuel sur cash (%)", min_value=0.0, value=5.0, step=0.5)

# 6. Paramètres de Fees
st.sidebar.markdown("### Fees")
management_fee = st.sidebar.number_input("Management Fee annuel (%)", min_value=0.0, value=2.0, step=0.1) / 100.0
performance_fee = st.sidebar.number_input("Performance Fee (%)", min_value=0.0, value=20.0, step=1.0) / 100.0

# 7. Période de forecast
forecast_days = st.sidebar.number_input("Période de forecast (jours)", min_value=30, value=365, step=30)

# 8. Stress Test
stress_shock = st.sidebar.number_input("Choc de stress (variation en %)", min_value=-100, max_value=0, value=-30, step=5)

st.markdown("## Dashboard Agrégé")

# -------------------------------
# Récupérer les données historiques et courantes
price_data = {}
current_prices = {}
for coin_id in selected_ids:
    df = fetch_crypto_history(coin_id, days=365)
    if not df.empty:
        price_data[coin_id] = df
        current_prices[coin_id] = df['price'].iloc[-1]

if not price_data:
    st.error("Aucune donnée historique récupérée pour les cryptos sélectionnées.")
    st.stop()

# Visualisation des courbes historiques pour chaque crypto
fig_hist = go.Figure()
for coin_id, df in price_data.items():
    coin_name = top_coins_df[top_coins_df['id'] == coin_id]['name'].values[0]
    fig_hist.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name=coin_name))
st.plotly_chart(fig_hist, use_container_width=True)

# Calcul de la valeur actuelle de chaque position en crypto selon l'allocation indiquée
crypto_values = {}
for coin_name, alloc_pct in allocation_inputs.items():
    coin_id = coin_dict[coin_name]
    alloc_amount = total_portfolio * (alloc_pct/100)
    crypto_values[coin_name] = alloc_amount  # On suppose achat au prix actuel

cash_value = total_portfolio * (cash_allocation/100)
portfolio_current = sum(crypto_values.values()) + cash_value
st.write(f"**Valeur totale actuelle du portefeuille :** ${portfolio_current:,.2f}")

# Dashboard Pie Chart de la répartition
fig_pie = portfolio_allocation_pie(crypto_values, cash_value)
st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------
# Projection & Forecast sur 1 an avec enveloppe de prix
st.markdown("## Projection sur 12 mois passés et 12 mois futurs")
projection_results = {}
projection_quantiles = {}
for coin_id in selected_ids:
    series = price_data[coin_id]['price']
    sim_df, quantiles = forecast_price_series(series, forecast_days=forecast_days, n_simulations=500)
    projection_results[coin_id] = sim_df
    projection_quantiles[coin_id] = quantiles

# Visualiser pour chaque crypto la projection avec enveloppe
for coin_id in selected_ids:
    coin_name = top_coins_df[top_coins_df['id'] == coin_id]['name'].values[0]
    quant_df = projection_quantiles[coin_id]
    fig_proj = go.Figure()
    # Enveloppe
    fig_proj.add_trace(go.Scatter(x=quant_df.index, y=quant_df['q05'], mode='lines', line=dict(color='red', dash='dash'), name='5e percentile'))
    fig_proj.add_trace(go.Scatter(x=quant_df.index, y=quant_df['q95'], mode='lines', line=dict(color='green', dash='dash'), name='95e percentile', fill='tonexty'))
    # Médiane
    fig_proj.add_trace(go.Scatter(x=quant_df.index, y=quant_df['q50'], mode='lines', line=dict(color='blue'), name='Médiane'))
    fig_proj.update_layout(title=f"Projection de Prix pour {coin_name} sur 12 mois",
                           xaxis_title="Date", yaxis_title="Prix ($)")
    st.plotly_chart(fig_proj, use_container_width=True)

# -------------------------------
# Analyses Complémentaires
st.markdown("## Analyses Complémentaires")

# 1. Scatter Plot Risque/Retour pour chaque crypto (annualisé)
risk_return_data = []
for coin_id, df in price_data.items():
    coin_name = top_coins_df[top_coins_df['id'] == coin_id]['name'].values[0]
    annual_returns = (df['price'].pct_change().dropna().mean() * 365)
    volatility, _ = risk_metrics(df['price'])
    risk_return_data.append({'Crypto': coin_name, 'Annual Return': annual_returns, 'Volatility': volatility})
risk_df = pd.DataFrame(risk_return_data)
fig_scatter = px.scatter(risk_df, x="Volatility", y="Annual Return", text="Crypto",
                         title="Scatter Plot Risque/Retour", labels={"Volatility": "Volatilité Annualisée", "Annual Return": "Retour Annualisé"})
st.plotly_chart(fig_scatter, use_container_width=True)

# 2. Heatmap de corrélation des rendements
# Construire un DataFrame avec les rendements quotidiens pour chaque crypto
price_df = pd.DataFrame()
for coin_id, df in price_data.items():
    coin_name = top_coins_df[top_coins_df['id'] == coin_id]['name'].values[0]
    price_df[coin_name] = df['price']
corr = price_df.pct_change().corr()
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Matrice de Corrélation des Rendements")
st.plotly_chart(fig_corr, use_container_width=True)
