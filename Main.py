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
    # On garde uniquement l'id, le symbole et le nom
    return df[['id', 'symbol', 'name']]

@st.cache_data(ttl=3600)
def fetch_crypto_history(coin_id, days=365, vs_currency='usd'):
    """
    Récupère l'historique des prix pour une crypto donnée via CoinGecko.
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

def forecast_price_series(price_series, forecast_days=30, n_simulations=100):
    """
    Effectue une simulation Monte-Carlo pour forecast le cours futur d'une crypto.
    On utilise ici une modélisation par mouvement Brownien géométrique.
    Retourne un DataFrame moyen des trajectoires simulées et l'ensemble des simulations.
    """
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()
    last_price = price_series.iloc[-1]

    dt = 1  # daily steps
    simulations = []

    for i in range(n_simulations):
        prices = [last_price]
        for d in range(forecast_days):
            # Génération d'un rendement aléatoire
            shock = np.random.normal(mu * dt, sigma * np.sqrt(dt))
            price = prices[-1] * np.exp(shock)
            prices.append(price)
        simulations.append(prices)

    sim_df = pd.DataFrame(simulations).T
    forecast_mean = sim_df.mean(axis=1)
    forecast_mean.index = [price_series.index[-1] + timedelta(days=i) for i in range(forecast_days+1)]
    return forecast_mean, sim_df

def apply_fees(portfolio_value, management_fee, performance_fee, initial_value):
    """
    Applique des frais de gestion et de performance sur la variation du portefeuille.
    Pour simplifier, on déduit le management fee en proportion linéaire sur la période, et
    le performance fee sur le gain (la partie positive).
    """
    # Management fee déduit linéairement sur la variation
    mg_fee = portfolio_value * management_fee
    gain = max(portfolio_value - initial_value, 0)
    perf_fee = gain * performance_fee
    net_value = portfolio_value - mg_fee - perf_fee
    return net_value

# -------------------------------
# Fonctions de Stress Test & Risk Analysis
# -------------------------------

def stress_test(portfolio_value_series, shock_percent):
    """
    Simule un stress test en appliquant un choc instantané sur la valeur du portefeuille.
    Retourne une série avec la valeur instantanée après choc.
    """
    shock_value = portfolio_value_series.iloc[-1] * (1 + shock_percent/100)
    stressed = portfolio_value_series.copy()
    stressed.iloc[-1] = shock_value
    return stressed

def risk_metrics(price_series):
    """
    Calcule quelques indicateurs de risque simples : volatilité (std), max drawdown.
    """
    returns = price_series.pct_change().dropna()
    volatility = returns.std()
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    return volatility, max_drawdown

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
                                          default=coin_options[:3])  # par défaut 3 cryptos

selected_ids = [coin_dict[name] for name in selected_coins]

# 2. Montant total du portefeuille et allocation
total_portfolio = st.sidebar.number_input("Montant total du portefeuille ($)", min_value=1000.0, value=100000.0, step=1000.0)

st.sidebar.markdown("### Allocation du portefeuille")
# Pour simplifier, on suppose une allocation égale entre les cryptos sélectionnées et le cash.
n_assets = len(selected_ids)
cash_allocation = st.sidebar.slider("Pourcentage d'allocation en cash", min_value=0, max_value=100, value=20, step=5)
crypto_allocation = 100 - cash_allocation

# Répartition égale entre les cryptos
if n_assets > 0:
    crypto_pct = crypto_allocation / n_assets
else:
    crypto_pct = 0

st.sidebar.markdown(f"Allocation crypto : {crypto_allocation}% (soit {crypto_pct:.1f}% par crypto)")

# 3. Stratégie de yield sur le cash
yield_cash = st.sidebar.number_input("Yield annuel sur cash (%)", min_value=0.0, value=5.0, step=0.5)

# 4. Fees
st.sidebar.markdown("### Paramètres de Fees")
management_fee = st.sidebar.number_input("Management Fee annuel (%)", min_value=0.0, value=2.0, step=0.1) / 100.0
performance_fee = st.sidebar.number_input("Performance Fee (%)", min_value=0.0, value=20.0, step=1.0) / 100.0

# 5. Forecast
forecast_days = st.sidebar.number_input("Période de forecast (jours)", min_value=7, value=30, step=1)

# 6. Stress Test
stress_shock = st.sidebar.number_input("Choc de stress (variation en %)", min_value=-100, max_value=0, value=-30, step=5)

st.markdown("## Visualisation des Données Historiques")

# Récupérer et afficher les données historiques pour chaque crypto sélectionnée
price_data = {}
current_prices = {}
for coin_id in selected_ids:
    df = fetch_crypto_history(coin_id, days=365)
    if not df.empty:
        price_data[coin_id] = df
        current_prices[coin_id] = df['price'].iloc[-1]

# Affichage des courbes pour chaque crypto
if price_data:
    fig = go.Figure()
    for coin_id, df in price_data.items():
        # Récupérer le nom pour le label
        coin_name = top_coins_df[top_coins_df['id'] == coin_id]['name'].values[0]
        fig.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name=coin_name))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Aucune donnée historique n'a pu être récupérée.")

# -------------------------------
# Calcul et Forecast du Portefeuille
# -------------------------------

st.markdown("## Forecast & Analyse du Portefeuille")

# Calculer la valeur actuelle du portefeuille
crypto_value = 0
for coin_id in selected_ids:
    allocation = total_portfolio * (crypto_allocation/100) * (1/n_assets)
    crypto_value += allocation  # on suppose que l'allocation correspond à l'achat à prix actuel
cash_value = total_portfolio * (cash_allocation/100)
portfolio_current = crypto_value + cash_value

st.write(f"**Valeur totale actuelle du portefeuille :** ${portfolio_current:,.2f}")

# Forecast pour chaque crypto (en utilisant les historiques)
forecast_results = {}
simulations = {}
for coin_id in selected_ids:
    if coin_id in price_data:
        series = price_data[coin_id]['price']
        forecast_mean, sim_df = forecast_price_series(series, forecast_days=forecast_days)
        forecast_results[coin_id] = forecast_mean
        simulations[coin_id] = sim_df

# Projection globale du portefeuille :
# On suppose que la valeur en crypto évolue selon le forecast et le cash génère du yield linéaire (annualisé)
portfolio_forecast = pd.DataFrame()
dates = None
if forecast_results:
    # Pour chaque crypto, calculer la valeur future proportionnelle à l'allocation initiale
    for coin_id, forecast_series in forecast_results.items():
        allocation = total_portfolio * (crypto_allocation/100) / n_assets
        # La valeur forecastée pour ce coin
        series_val = forecast_series * (allocation / current_prices[coin_id])
        if portfolio_forecast.empty:
            portfolio_forecast = series_val.to_frame(name=coin_id)
            dates = series_val.index
        else:
            portfolio_forecast[coin_id] = series_val
    # Somme des valeurs crypto
    portfolio_forecast['Crypto_Total'] = portfolio_forecast.sum(axis=1)
else:
    st.warning("Forecast indisponible pour les cryptos.")

# Projection du cash avec yield (simple proportionnel au temps)
forecast_cash = cash_value * (1 + (yield_cash/100) * (np.array(range(forecast_days+1))/365))
forecast_cash_series = pd.Series(forecast_cash, index=dates if dates is not None else pd.date_range(datetime.now(), periods=forecast_days+1))
portfolio_forecast['Cash'] = forecast_cash_series

portfolio_forecast['Total'] = portfolio_forecast.sum(axis=1)

# Application des fees sur la valeur finale forecastée (pour simplifier, appliquée sur le gain par rapport à la valeur initiale)
initial_value = portfolio_current
portfolio_forecast['Net_Total'] = portfolio_forecast['Total'].apply(lambda x: apply_fees(x, management_fee, performance_fee, initial_value))

st.subheader("Forecast du Portefeuille sur la Période")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=portfolio_forecast.index, y=portfolio_forecast['Total'], mode='lines', name='Total Brut'))
fig2.add_trace(go.Scatter(x=portfolio_forecast.index, y=portfolio_forecast['Net_Total'], mode='lines', name='Total Net (après fees)'))
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Stress Test & Risk Metrics
# -------------------------------

st.markdown("## Stress Test & Analyse de Risque")
# Stress Test : appliquer un choc sur la dernière valeur forecastée
stressed_series = stress_test(portfolio_forecast['Total'], shock_percent=stress_shock)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=portfolio_forecast.index, y=portfolio_forecast['Total'], mode='lines', name='Forecast Normal'))
fig3.add_trace(go.Scatter(x=portfolio_forecast.index, y=stressed_series, mode='lines', name=f'Stress Test ({stress_shock}%)'))
st.plotly_chart(fig3, use_container_width=True)

# Calcul des indicateurs de risque pour le portefeuille (uniquement sur la série forecastée brute)
volatility, max_drawdown = risk_metrics(portfolio_forecast['Total'])
st.write(f"**Volatilité forecastée (std) :** {volatility:.2%}")
st.write(f"**Max Drawdown forecasté :** {max_drawdown:.2%}")

# -------------------------------
# Projection empirique sur le Yield par Asset
# -------------------------------
st.markdown("## Projection Empirique des Yields par Actif")
for coin_id in selected_ids:
    coin_name = top_coins_df[top_coins_df['id'] == coin_id]['name'].values[0]
    st.write(f"### {coin_name}")
    # Pour illustrer, on suppose que le yield sur l'actif est une fraction du rendement annualisé historique
    series = price_data[coin_id]['price']
    log_returns = np.log(series/series.shift(1)).dropna()
    annual_return = np.exp(log_returns.mean()*365) - 1
    # Par exemple, on considère 20% de ce rendement comme yield (valeur indicative)
    asset_yield = annual_return * 0.2
    st.write(f"Rendement annualisé historique : {annual_return:.2%}")
    st.write(f"Projection de yield (20% du rendement historique) : {asset_yield:.2%}")
    # Visualisation simple de l'historique de l'actif
    fig_asset = px.line(series.reset_index(), x='timestamp', y='price', title=f"Historique du Prix pour {coin_name}")
    st.plotly_chart(fig_asset, use_container_width=True)
