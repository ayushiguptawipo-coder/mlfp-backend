import os
import requests
import pandas as pd
import numpy as np
import re
import json
import urllib.parse
import feedparser
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from google import genai
from google.genai import types
from datetime import datetime, timedelta

# SECURE ENVIRONMENT VARIABLES (Pulled directly from Render)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
UPSTOX_ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN")

app = FastAPI(title="MLFP Quant Engine API")

# Allow Web Dashboard Frontend Communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRICTION_COST = 0.0015
UPSTOX_KEYS = {
    'SBIN': 'NSE_EQ|INE062A01020',
    'RELIANCE': 'NSE_EQ|INE002A01018',
    'INFY': 'NSE_EQ|INE009A01021'
}

def fetch_upstox_data(symbol, years=2):
    instrument_key = UPSTOX_KEYS.get(symbol)
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    url = f'https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{to_date}/{from_date}'
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json().get('data', {}).get('candles', [])
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'ClosePrice', 'Volume', 'OI'])
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        return df.sort_values('Date').reset_index(drop=True)[['Date', 'ClosePrice', 'Volume']]
    return pd.DataFrame()

def generate_hybrid_features(df):
    df = df.copy()
    df['Log_Returns'] = np.log(df['ClosePrice'] / df['ClosePrice'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Forward_Return'] = np.log(df['ClosePrice'].shift(-1) / df['ClosePrice'])
    
    sma_20_raw = df['ClosePrice'].rolling(window=20).mean()
    df['SMA_20_Dist'] = (df['ClosePrice'] - sma_20_raw) / sma_20_raw
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    delta = df['ClosePrice'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / loss)))
    
    for col in ['Log_Returns', 'RSI_14']:
        df[f'{col}_Lag1'] = df[col].shift(1)
        df[f'{col}_Lag2'] = df[col].shift(2)
        
    ar_preds, returns_series = [], df['Log_Returns'].values
    for i in range(len(df)):
        if i < 20: ar_preds.append(np.nan)
        else:
            window = returns_series[max(0, i-30):i]
            X_ar, y_ar = window[:-1].reshape(-1, 1), window[1:]
            if len(X_ar) > 5 and not np.isnan(X_ar).any() and not np.isnan(y_ar).any():
                reg = Ridge(alpha=1.0).fit(X_ar, y_ar)
                ar_preds.append(reg.predict(window[-1].reshape(1, -1))[0])
            else: ar_preds.append(0.0)
                
    df['AR1_Forecast'] = ar_preds
    df['Target_Direction'] = (df['Forward_Return'] > 0).astype(int)
    df['Rolling_Vol'] = df['Log_Returns'].rolling(window=10).std()
    return df.dropna().reset_index(drop=True)

@app.get("/")
def home():
    return {"status": "MLFP Quant Engine API is online and running!"}

@app.get("/api/analyze/{ticker}")
def analyze_stock(ticker: str):
    ticker = ticker.upper()
    if ticker not in UPSTOX_KEYS:
        raise HTTPException(status_code=400, detail="Ticker not supported. Choose SBIN, RELIANCE, or INFY.")
    
    if not GEMINI_API_KEY or not UPSTOX_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="API Keys are missing on server configuration.")

    pooled_data = [generate_hybrid_features(fetch_upstox_data(t)) for t in UPSTOX_KEYS.keys() if not fetch_upstox_data(t).empty]
    if not pooled_data: raise HTTPException(status_code=500, detail="Failed to fetch Upstox data.")
    master_df = pd.concat(pooled_data, ignore_index=True).sort_values('Date').reset_index(drop=True)

    log_vol = np.log(master_df[['Rolling_Vol']] + 1e-8)
    scaled_vol = StandardScaler().fit_transform(log_vol)
    master_df['Volatility_Regime'] = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(scaled_vol)
    regime_means = master_df.groupby('Volatility_Regime')['Rolling_Vol'].mean()
    master_df['Regime_Label'] = master_df['Volatility_Regime'].apply(
        lambda x: 'Low Volatility' if x == regime_means.idxmin() else 'High Volatility'
    )

    feature_cols = ['Log_Returns', 'SMA_20_Dist', 'RSI_14', 'Relative_Volume', 'Log_Returns_Lag1', 'Log_Returns_Lag2', 'RSI_14_Lag1', 'AR1_Forecast']
    X, y = master_df[feature_cols], master_df['Target_Direction']
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
    }
    
    results = {name: {'auc': []} for name in models.keys()}
    tscv = TimeSeriesSplit(n_splits=5)
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            try: results[name]['auc'].append(roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))
            except: results[name]['auc'].append(0.5)

    best_model_name = max(results, key=lambda k: np.mean(results[k]['auc']))
    
    oos_preds, oos_indices = [], []
    for train_idx, test_idx in tscv.split(X):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X.iloc[train_idx])
        X_test_scaled = scaler.transform(X.iloc[test_idx])
        model = models[best_model_name].fit(X_train_scaled, y.iloc[train_idx])
        oos_preds.extend(model.predict(X_test_scaled))
        oos_indices.extend(test_idx)

    res_df = master_df.iloc[oos_indices].copy()
    res_df['OOS_Pred'] = oos_preds
    res_df['Friction_Unfilt'] = (res_df['OOS_Pred'].diff().abs().fillna(0)) * FRICTION_COST
    res_df['Ret_Unfilt'] = (res_df['OOS_Pred'] * res_df['Forward_Return']) - res_df['Friction_Unfilt']
    
    res_df['Pos_Filt'] = np.where(res_df['Regime_Label'] == 'Low Volatility', res_df['OOS_Pred'], 0)
    res_df['Friction_Filt'] = (res_df['Pos_Filt'].diff().abs().fillna(0)) * FRICTION_COST
    res_df['Ret_Filt'] = (res_df['Pos_Filt'] * res_df['Forward_Return']) - res_df['Friction_Filt']

    clean_df = res_df.dropna(subset=['Ret_Unfilt', 'Ret_Filt', 'Forward_Return'])
    
    target_df = generate_hybrid_features(fetch_upstox_data(ticker))
    vol_scaled_target = StandardScaler().fit(log_vol).transform(np.log(target_df[['Rolling_Vol']] + 1e-8))
    target_df['Volatility_Regime'] = KMeans(n_clusters=2, random_state=42, n_init=10).fit(scaled_vol).predict(vol_scaled_target)
    current_regime = 'Low Volatility' if target_df['Volatility_Regime'].iloc[-1] == regime_means.idxmin() else 'High Volatility'
    
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    live_model = models[best_model_name].fit(X_scaled, y)
    prob_up = live_model.predict_proba(final_scaler.transform(target_df[feature_cols].iloc[[-1]]))[0][1]

    # --- GEMINI API ENGINE ---
    ai_score, ai_summary = 0.0, "AI Sentiment Unavailable"
    try:
        q = urllib.parse.quote(f"{ticker} stock news India")
        rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        rss_resp = requests.get(rss_url, headers=headers)
        feed = feedparser.parse(rss_resp.content)
        
        headlines = "\n".join([f"- {h.title}" for h in feed.entries[:10]])
        
        if not headlines.strip():
            raise ValueError("No headlines found.")

        client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(initial_delay=1.0, attempts=2)))
        prompt = f"Analyze these recent news headlines for '{ticker}':\n{headlines}\nReturn ONLY a valid JSON: {{\"sentiment_score\": <float -1.0 to 1.0>, \"executive_summary\": \"<1 sentence>\"}}"
        
        resp = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt, 
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        match = re.search(r'\{.*\}', resp.text, re.DOTALL)
        ai_json = json.loads(match.group(0)) if match else json.loads(resp.text)
        ai_score = ai_json.get('sentiment_score', 0)
        ai_summary = ai_json.get('executive_summary', "No summary provided.")
    except Exception as e:
        ai_summary = f"API Diagnostic: {str(e)}"

    return {
        "ticker": ticker,
        "current_regime": current_regime,
        "best_model": best_model_name,
        "quant_signal": "BULLISH" if prob_up >= 0.5 else "BEARISH",
        "quant_probability": round(prob_up * 100, 2),
        "ai_sentiment_score": ai_score,
        "ai_summary": ai_summary,
        "diagnostics": {
            "unfiltered_trades": int(clean_df['OOS_Pred'].diff().abs().sum()),
            "filtered_trades": int(clean_df['Pos_Filt'].diff().abs().sum()),
            "unfiltered_friction_pct": round(clean_df['Friction_Unfilt'].sum() * 100, 2),
            "filtered_friction_pct": round(clean_df['Friction_Filt'].sum() * 100, 2)
        },
        "performance": {
            "buy_and_hold_pct": round((np.exp(clean_df['Forward_Return'].sum()) - 1) * 100, 2),
            "unfiltered_strat_pct": round((np.exp(clean_df['Ret_Unfilt'].sum()) - 1) * 100, 2),
            "filtered_strat_pct": round((np.exp(clean_df['Ret_Filt'].sum()) - 1) * 100, 2)
        },
        "chart_data": {
            "labels": [str(d.date()) for d in clean_df['Date']],
            "buy_hold": [round(x, 4) for x in list(np.exp(clean_df['Forward_Return'].cumsum()) - 1)],
            "unfiltered": [round(x, 4) for x in list(np.exp(clean_df['Ret_Unfilt'].cumsum()) - 1)],
            "filtered": [round(x, 4) for x in list(np.exp(clean_df['Ret_Filt'].cumsum()) - 1)]
        }
    }
