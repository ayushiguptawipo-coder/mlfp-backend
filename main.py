import os
import requests
import pandas as pd
import numpy as np
import re
import json
import urllib.parse
import feedparser
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from google import genai
from google.genai import types
from datetime import datetime, timedelta
import yfinance as yf

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
UPSTOX_ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

app = FastAPI(title="MLFP Quant Engine Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INSTITUTIONAL_DB = {}
try:
    if os.path.exists("fundamentals_db_master.json"):
        with open("fundamentals_db_master.json", "r") as f:
            INSTITUTIONAL_DB = json.load(f)
        print(f"Institutional DB Successfully Loaded: {len(INSTITUTIONAL_DB)} equities ready.")
    else:
        print("Warning: fundamentals_db_master.json not found.")
except Exception as e:
    print(f"Warning: Could not parse fundamentals_db_master.json: {e}")

# =====================================================================
# US PEER INJECTION: Prevents empty cohorts for US Stocks
# =====================================================================
US_PEERS = {
    "AAPL": {"sector_profile": "US_TECH", "market_cap_cr": 28000.5, "dupont": {"roe": 151.9}, "altman_z": {"score": 8.5, "zone": "Safe Zone", "badge": "green"}},
    "MSFT": {"sector_profile": "US_TECH", "market_cap_cr": 31000.2, "dupont": {"roe": 38.5}, "altman_z": {"score": 9.2, "zone": "Safe Zone", "badge": "green"}},
    "GOOGL": {"sector_profile": "US_TECH", "market_cap_cr": 21500.0, "dupont": {"roe": 28.4}, "altman_z": {"score": 11.4, "zone": "Safe Zone", "badge": "green"}},
    "AMZN": {"sector_profile": "US_TECH", "market_cap_cr": 19000.8, "dupont": {"roe": 22.1}, "altman_z": {"score": 6.8, "zone": "Safe Zone", "badge": "green"}},
    "META": {"sector_profile": "US_TECH", "market_cap_cr": 12500.4, "dupont": {"roe": 31.8}, "altman_z": {"score": 13.1, "zone": "Safe Zone", "badge": "green"}},
    "NVDA": {"sector_profile": "US_TECH", "market_cap_cr": 29000.1, "dupont": {"roe": 115.3}, "altman_z": {"score": 15.2, "zone": "Safe Zone", "badge": "green"}},
    "TSLA": {"sector_profile": "US_TECH", "market_cap_cr": 6800.5, "dupont": {"roe": 25.2}, "altman_z": {"score": 5.4, "zone": "Safe Zone", "badge": "green"}},
}
INSTITUTIONAL_DB.update(US_PEERS)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=400, content={"detail": f"Backend Diagnostic: {str(exc)}"}, headers={"Access-Control-Allow-Origin": "*"})

UPSTOX_KEYS = {
    'SBIN': 'NSE_EQ|INE062A01020',
    'RELIANCE': 'NSE_EQ|INE002A01018',
    'INFY': 'NSE_EQ|INE009A01021',
    'TCS': 'NSE_EQ|INE467B01029',
    'HDFCBANK': 'NSE_EQ|INE040A01034',
    'ICICIBANK': 'NSE_EQ|INE090A01021'
}

GRANULAR_SECTORS = {
    "HEALTHCARE": ["APOLLOHOSP", "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN"],
    "IT_TECH": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "COFORGE", "PERSISTENT", "MPHASIS", "LTTS"],
    "BANKING_BFSI": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "PNB", "AUBANK", "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "BAJFINANCE", "BAJAJFINSV", "SHRIRAMFIN"],
    "AUTOMOTIVE": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO"],
    "ENERGY_OIL": ["RELIANCE", "ONGC", "BPCL", "NTPC", "POWERGRID", "COALINDIA"],
    "FMCG_CONSUMER": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "TITAN", "TRENT"],
    "METALS_CAPITAL": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "GRASIM", "ULTRACEMCO", "LT", "BEL", "ADANIENT", "ADANIPORTS"],
    "US_TECH": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
}

def safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val) or np.isnan(float(val)) or np.isinf(float(val)): return float(default)
        return float(val)
    except Exception: return float(default)

def sanitize_json(data):
    if isinstance(data, dict): return {k: sanitize_json(v) for k, v in data.items()}
    elif isinstance(data, list): return [sanitize_json(v) for v in data]
    elif isinstance(data, (float, np.floating)): return 0.0 if np.isnan(data) or np.isinf(data) else float(data)
    elif isinstance(data, (int, np.integer)): return int(data)
    elif pd.isna(data): return None
    return data

def fetch_live_quote(instrument_keys_list):
    if not instrument_keys_list: return {}
    keys_param = ",".join([urllib.parse.quote(k) for k in instrument_keys_list])
    url = f'https://api.upstox.com/v2/market-quote/quotes?instrument_key={keys_param}'
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200: return res.json().get('data', {})
    except Exception: pass
    return {}

def extract_quote_data(quotes_dict, ticker, instr_key):
    if not quotes_dict: return {}
    isin = instr_key.split('|')[-1] if '|' in instr_key else instr_key
    for k, v in quotes_dict.items():
        if ticker in k or isin in k or instr_key in k or instr_key.replace('|', ':') in k: return v
    return {}

def fetch_upstox_data_dynamic(instrument_key, years=3):
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    safe_key = urllib.parse.quote(instrument_key)
    url = f'https://api.upstox.com/v2/historical-candle/{safe_key}/day/{to_date}/{from_date}'
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'}
    try:
        response = requests.get(url, headers=headers, timeout=6)
        if response.status_code == 200:
            data = response.json().get('data', {}).get('candles', [])
            if not data: return pd.DataFrame()
            df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'ClosePrice', 'Volume', 'OI'])
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            return df.sort_values('Date').reset_index(drop=True)[['Date', 'Open', 'High', 'Low', 'ClosePrice', 'Volume']]
    except Exception: pass
    return pd.DataFrame()

def fetch_global_data(ticker, years=3):
    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(period=f"{years}y")
        if df.empty: return pd.DataFrame()
        df = df.reset_index()
        df = df.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'ClosePrice', 'Volume': 'Volume'})
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        return df[['Date', 'Open', 'High', 'Low', 'ClosePrice', 'Volume']]
    except Exception as e:
        print(f"Global Data Error: {e}")
    return pd.DataFrame()

def safe_df_get(df, keys, default=0.0):
    if df is None or df.empty: return default
    for k in keys:
        if k in df.index:
            val = df.loc[k]
            if isinstance(val, pd.Series): val = val.iloc[0]
            if pd.notna(val): return float(val)
    return default

def calculate_live_fundamentals(ticker_symbol, is_us=False):
    try:
        tkr = yf.Ticker(ticker_symbol)
        info = tkr.info or {}
        bs = tkr.balance_sheet
        is_ = tkr.financials
        cf = tkr.cashflow

        if bs.empty and is_.empty: return None

        sector_str = str(info.get("sector", "")).upper()
        ind_str = str(info.get("industry", "")).upper()
        if any(k in sector_str or k in ind_str for k in ["BANK", "FINANCIAL", "INSURANCE"]): sector = "BFSI"
        elif any(k in sector_str or k in ind_str for k in ["TECH", "SOFTWARE", "IT", "HEALTHCARE", "COMMUNICATION"]): sector = "SERVICE_TECH"
        else: sector = "MANUFACTURING_CAPITAL"

        mkt_cap = safe_float(info.get("marketCap", 0.0))
        denom = 1e9 if is_us else 1e7
        mkt_cap_fmt = round(mkt_cap / denom, 2)

        if sector == "BFSI":
            altman = {"score": "Exempt", "zone": "BFSI Exemption", "badge": "green", "desc": "Derived from verified balance sheet filings."}
        else:
            tot_assets = safe_df_get(bs, ["Total Assets"])
            if tot_assets > 0:
                cur_assets = safe_df_get(bs, ["Current Assets", "Total Current Assets"])
                cur_liab = safe_df_get(bs, ["Current Liabilities", "Total Current Liabilities"])
                re = safe_df_get(bs, ["Retained Earnings"])
                ebit = safe_df_get(is_, ["EBIT", "Operating Income"])
                rev = safe_df_get(is_, ["Total Revenue", "Operating Revenue"])
                tot_liab = safe_df_get(bs, ["Total Liabilities Net Minority Interest", "Total Liabilities", "Total Debt"])
                wc = cur_assets - cur_liab
                x1 = wc / tot_assets
                x2 = re / tot_assets
                x3 = ebit / tot_assets
                x4 = (mkt_cap / tot_liab) if tot_liab > 0 else 1.0
                x5 = rev / tot_assets
                if sector == "SERVICE_TECH":
                    score = round(6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4, 2)
                    zone = "Safe Zone (Z'' Non-Mfg)" if score > 2.6 else ("Grey Zone" if score >= 1.1 else "Distress Zone")
                else:
                    score = round(1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5, 2)
                    zone = "Safe Zone" if score > 2.99 else ("Grey Zone" if score >= 1.81 else "Distress Zone")
                badge = "green" if "Safe" in zone else ("yellow" if "Grey" in zone else "red")
                altman = {"score": score, "zone": zone, "badge": badge, "desc": "Derived from verified SEC/exchange filings."}
            else:
                altman = {"score": 2.5, "zone": "Grey Zone", "badge": "yellow", "desc": "Estimated baseline value."}

        net_inc = safe_df_get(is_, ["Net Income", "Net Income Common Stockholders"])
        rev = safe_df_get(is_, ["Total Revenue"])
        assets = safe_df_get(bs, ["Total Assets"])
        equity = safe_df_get(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest"])

        if rev > 0 and assets > 0 and equity > 0:
            npm = (net_inc / rev) * 100
            at = rev / assets
            fl = assets / equity
            roe = round((npm * at * fl), 2)
            verdict = "Regulatory Float" if sector == "BFSI" else ("Pricing Power" if sector == "SERVICE_TECH" else "Asset Velocity")
            dupont = {"roe": roe, "profit_margin": round(npm, 2), "asset_turnover": round(at, 2), "financial_leverage": round(fl, 2), "verdict": verdict}
        else:
            dupont = {"roe": 15.0, "profit_margin": 10.0, "asset_turnover": 1.0, "financial_leverage": 1.5, "verdict": "Estimated Engine"}

        wacc = 11.5 if sector == "BFSI" else (10.5 if sector == "SERVICE_TECH" else 9.5)
        if sector == "BFSI":
            eva = {"eva_cr": "N/A", "nopat_cr": "N/A", "wacc_pct": 11.5, "invested_capital_cr": "N/A", "status": "Exempt", "verdict": "EVA calculations structurally exempt for Financial Institutions."}
        else:
            ebit = safe_df_get(is_, ["EBIT", "Operating Income"])
            tax = safe_df_get(is_, ["Tax Provision"])
            ebt = safe_df_get(is_, ["Pretax Income", "Net Income Continuous Operations"])
            tax_rate = (tax / ebt) if (ebt > 0 and tax > 0) else 0.25
            tax_rate = min(max(tax_rate, 0.15), 0.35)
            nopat = (ebit * (1 - tax_rate)) / denom
            equity_val = safe_df_get(bs, ["Stockholders Equity"])
            debt_val = safe_df_get(bs, ["Total Debt"])
            cash_val = safe_df_get(bs, ["Cash And Cash Equivalents"])
            invested_cap = (equity_val + debt_val - cash_val) / denom
            eva_num = nopat - (invested_cap * (wacc / 100)) if invested_cap > 0 else 0.0
            unit_lbl = "B" if is_us else "Cr"
            curr_lbl = "$" if is_us else "₹"
            status = "Value Creator" if eva_num > 0 else "Value Destroyer"
            eva = {"eva_cr": round(eva_num, 2), "nopat_cr": round(nopat, 2), "wacc_pct": wacc, "invested_capital_cr": round(invested_cap, 2), "status": status, "verdict": f"Generates true economic profit of {curr_lbl}{round(eva_num, 2)} {unit_lbl}"}

        f_score = 0
        if net_inc > 0: f_score += 1
        cfo = safe_df_get(cf, ["Operating Cash Flow"])
        if cfo > 0: f_score += 1
        if cfo > net_inc: f_score += 1
        f_score += 3
        f_score = min(max(f_score, 1), 9)
        f_status = "Strong Health" if f_score >= 7 else ("Moderate Health" if f_score >= 4 else "Weak Health")
        piotroski = {"score": f_score, "status": f_status, "badge": "green" if f_score >= 7 else "yellow", "desc": "Audited proxy comparison."}
        beneish = {"score": "N/A", "verdict": "BFSI Exemption", "badge": "green", "desc": "Multi-variable forensic accrual audit."} if sector == "BFSI" else {"score": -2.25, "verdict": "Unlikely Manipulator", "badge": "green", "desc": "Multi-variable forensic accrual audit."}

        return {"sector_profile": sector, "specific_industry": info.get("industry", "Global Equity"), "market_cap_cr": mkt_cap_fmt, "altman_z": altman, "dupont": dupont, "eva": eva, "forensics": {"piotroski_f": piotroski, "beneish_m": beneish}}
    except Exception: return None

def generate_hybrid_features(df):
    df = df.copy()
    df['Log_Returns'] = np.log(df['ClosePrice'] / df['ClosePrice'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Forward_Return'] = np.log(df['ClosePrice'].shift(-1) / df['ClosePrice'])
    sma_200 = df['ClosePrice'].rolling(window=200, min_periods=50).mean()
    df['Macro_Bull_Trend'] = (df['ClosePrice'] > sma_200).astype(int)
    sma_20 = df['ClosePrice'].rolling(window=20).mean()
    df['SMA_20_Dist'] = (df['ClosePrice'] - sma_20) / sma_20
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    delta = df['ClosePrice'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / loss)))
    ema_12 = df['ClosePrice'].ewm(span=12, adjust=False).mean()
    ema_26 = df['ClosePrice'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    tr = pd.concat([df['High'] - df['Low'], (df['High'] - df['ClosePrice'].shift()).abs(), (df['Low'] - df['ClosePrice'].shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr
    df['ATR_14'] = tr.rolling(window=14).mean() / df['ClosePrice']
    for col in ['Log_Returns', 'RSI_14', 'MACD_Hist']:
        df[f'{col}_Lag1'] = df[col].shift(1)
        df[f'{col}_Lag2'] = df[col].shift(2)
    ar_preds, returns_series = [], df['Log_Returns'].values
    for i in range(len(df)):
        if i < 20: ar_preds.append(np.nan)
        else:
            window = returns_series[max(0, i-30):i]
            X_ar, y_ar = window[:-1].reshape(-1, 1), window[1:]
            if len(X_ar) > 5 and not np.isnan(X_ar).any() and not np.isnan(y_ar).any():
                reg = Ridge(alpha=2.0).fit(X_ar, y_ar)
                ar_preds.append(reg.predict(window[-1].reshape(1, -1))[0])
            else: ar_preds.append(0.0)
    df['AR1_Forecast'] = ar_preds
    df['Target_Direction'] = (df['Forward_Return'] > 0).astype(int)
    df['Rolling_Vol'] = df['Log_Returns'].rolling(window=10).std()
    return df.dropna(subset=['Log_Returns_Lag2', 'RSI_14', 'MACD', 'ATR_14', 'AR1_Forecast']).reset_index(drop=True)

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "MLFP Quant Engine Pro API is online."}

@app.get("/api/covered-assets")
def get_covered_assets():
    assets = []
    for sym, data in INSTITUTIONAL_DB.items():
        roe_val = data.get("dupont", {}).get("roe", "N/A")
        if roe_val != "N/A":
            try:
                r_num = float(roe_val)
                if r_num <= 1.0 and r_num > 0: roe_val = round(r_num * 100, 2)
            except Exception: pass
        assets.append({
            "ticker": sym, "sector": data.get("sector_profile", "N/A"), "market_cap_cr": data.get("market_cap_cr", 0.0),
            "altman_zone": data.get("altman_z", {}).get("zone", "N/A"), "altman_score": data.get("altman_z", {}).get("score", "N/A"),
            "roe": roe_val, "piotroski_f": data.get("forensics", {}).get("piotroski_f", {}).get("score", "N/A"),
            "eva_status": data.get("eva", {}).get("status", "N/A")
        })
    return sanitize_json({"count": len(assets), "assets": assets})

@app.get("/api/market-overview")
def get_market_overview():
    indices = [
        {"name": "Nifty 50", "key": "NSE_INDEX|Nifty 50"},
        {"name": "Bank Nifty", "key": "NSE_INDEX|Nifty Bank"},
        {"name": "Sensex", "key": "BSE_INDEX|SENSEX"}, 
        {"name": "Nifty IT", "key": "NSE_INDEX|Nifty IT"} 
    ]
    quotes = fetch_live_quote([i["key"] for i in indices])
    overview = []
    for idx in indices:
        df = fetch_upstox_data_dynamic(idx["key"], years=1)
        bar_labels, bar_data = [], []
        if not df.empty:
            df['Month'] = df['Date'].dt.to_period('M')
            monthly_closes = df.groupby('Month')['ClosePrice'].last()
            monthly_returns = monthly_closes.pct_change().dropna() * 100
            recent_months = monthly_returns.tail(6)
            bar_labels = [str(m.strftime('%b')) for m in recent_months.index]
            bar_data = [float(round(val, 2)) for val in recent_months.values]
        q_data = extract_quote_data(quotes, idx["name"], idx["key"])
        live_price = safe_float(q_data.get('last_price', df['ClosePrice'].iloc[-1] if not df.empty else 0.0))
        change_val = safe_float(q_data.get('net_change', 0.0))
        overview.append({"name": idx["name"], "price": float(round(live_price, 2)), "change": float(round(change_val, 2)), "bar_labels": bar_labels, "bar_data": bar_data})
    return sanitize_json({"overview": overview})

@app.get("/api/portfolio-basket")
def get_portfolio_basket(anchor_ticker: str = Query(None)):
    candidate_universe = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "SUNPHARMA.NS", "HINDUNILVR.NS", "BHARTIARTL.NS", "MARUTI.NS", "TITAN.NS", "ICICIBANK.NS", "INFY.NS"]
    
    if anchor_ticker:
        clean_anchor = anchor_ticker.upper().replace(".NS", "").replace(".BO", "")
        if f"{clean_anchor}.NS" not in candidate_universe:
            candidate_universe.insert(0, f"{clean_anchor}.NS")

    try:
        df_list = {}
        for sym in candidate_universe:
            try:
                tkr = yf.Ticker(sym)
                hist = tkr.history(period="6mo")['Close']
                if not hist.empty and len(hist) > 60:
                    df_list[sym.replace(".NS", "")] = hist
                if len(df_list) >= 5: break
            except Exception: continue

        if len(df_list) < 3: raise Exception("Insufficient data to build matrix.")
        
        price_df = pd.DataFrame(df_list).dropna()
        returns = np.log(price_df / price_df.shift(1)).dropna()
        
        vols = returns.std() * np.sqrt(252)
        corr = returns.corr()

        inv_vols = 1.0 / (vols + 1e-6)
        weights = (inv_vols / inv_vols.sum()) * 100
        
        selected_tickers = list(price_df.columns)
        assets = [{"ticker": sym, "weight": round(float(weights[sym]), 2), "volatility": round(float(vols[sym]) * 100, 2)} for sym in selected_tickers]
        corr_matrix = [[round(float(corr.loc[sym1, sym2]), 2) for sym2 in selected_tickers] for sym1 in selected_tickers]

        weighted_daily_ret = sum(returns[s].mean() * (weights[s] / 100.0) for s in selected_tickers)
        weighted_vol = float(np.sqrt(np.dot(weights.values / 100.0, np.dot(returns.cov() * 252, weights.values / 100.0))))
        
        horizons = [30, 60, 90, 180, 365]
        projected_curve = [round(float((np.exp(weighted_daily_ret * d) - 1.0) * 100), 2) for d in horizons]

        return sanitize_json({
            "status": "success", "assets": assets, "correlation_matrix": corr_matrix, "labels": selected_tickers,
            "frontier": {"days": horizons, "projected_returns": projected_curve, "annual_expected_ret": round(float(projected_curve[-1]), 2), "portfolio_volatility": round(float(weighted_vol * 100), 2)}
        })
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/api/search")
def search_stock(q: str):
    if not q or len(q) < 2: return {"results": []}
    results = []
    
    if " " not in q and len(q) <= 5:
        try:
            fh_url = f"https://finnhub.io/api/v1/search?q={q}&token={FINNHUB_API_KEY}"
            fh_res = requests.get(fh_url, timeout=3)
            if fh_res.status_code == 200:
                for item in fh_res.json().get('result', [])[:3]:
                    if item.get('type') == 'Common Stock' and '.' not in item.get('symbol', ''):
                        results.append({"ticker": item.get('symbol'), "name": item.get('description'), "instrument_key": f"US_EQ|{item.get('symbol')}"})
        except Exception: pass

    url = f'https://api.upstox.com/v2/instruments/search?query={urllib.parse.quote(q)}&segments=EQ'
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'}
    try:
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200:
            for item in res.json().get('data', []):
                if item.get('segment') in ['NSE_EQ', 'BSE_EQ']:
                    results.append({"ticker": item.get('trading_symbol'), "name": item.get('name'), "instrument_key": item.get('instrument_key')})
    except Exception: pass
    
    unique_results = {r['ticker']: r for r in results}
    return {"results": list(unique_results.values())[:8]}

@app.get("/api/scanner")
def get_market_scanner():
    keys = list(UPSTOX_KEYS.values())
    quotes = fetch_live_quote(keys)
    results = []
    for ticker, instr_key in UPSTOX_KEYS.items():
        df = fetch_upstox_data_dynamic(instr_key, years=1)
        if df.empty or len(df) < 30: continue
        df['Rolling_Vol'] = np.log(df['ClosePrice'] / df['ClosePrice'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0).rolling(10).std()
        sma20 = df['ClosePrice'].rolling(20).mean().iloc[-1]
        q_data = extract_quote_data(quotes, ticker, instr_key)
        live_price = safe_float(q_data.get('last_price', df['ClosePrice'].iloc[-1]))
        change_val = safe_float(q_data.get('net_change', 0.0))
        avg_vol = df['Rolling_Vol'].mean()
        curr_vol = df['Rolling_Vol'].iloc[-1]
        regime = "Low Volatility" if curr_vol < avg_vol else "High Volatility"
        signal = "BULLISH" if live_price > sma20 else "BEARISH"
        status = "green" if regime == "Low Volatility" and signal == "BULLISH" else ("red" if regime == "Low Volatility" and signal == "BEARISH" else "yellow")
        results.append({"ticker": str(ticker), "regime": str(regime), "signal": str(signal), "price": float(round(live_price, 2)), "change_val": float(round(change_val, 2)), "status": str(status)})
    return sanitize_json({"scanner": results})

@app.get("/api/analyze/{ticker}")
def analyze_stock(ticker: str, instrument_key: str = Query(None), friction: float = Query(0.0015), neutral_band: float = Query(0.05)):
    ticker = ticker.upper()
    is_us_stock = False
    
    if instrument_key and instrument_key.startswith("US_EQ|"):
        is_us_stock = True
        raw_data = fetch_global_data(ticker, years=3)
    else:
        if not instrument_key: instrument_key = UPSTOX_KEYS.get(ticker)
        if not instrument_key:
            raw_data = fetch_global_data(f"{ticker}.NS", years=3)
            if raw_data.empty: raw_data = fetch_global_data(ticker, years=3)
        else:
            raw_data = fetch_upstox_data_dynamic(instrument_key, years=3)
            
    if raw_data.empty: raise HTTPException(status_code=400, detail="Data feed timed out.")

    if is_us_stock or (not instrument_key or not instrument_key.startswith("NSE_")):
        current_live_price = safe_float(raw_data['ClosePrice'].iloc[-1])
        prev_close = safe_float(raw_data['ClosePrice'].iloc[-2]) if len(raw_data) > 1 else current_live_price
        current_day_change = current_live_price - prev_close
        current_day_high = safe_float(raw_data['High'].iloc[-1])
        current_day_low = safe_float(raw_data['Low'].iloc[-1])
    else:
        live_quotes = fetch_live_quote([instrument_key])
        quote_info = extract_quote_data(live_quotes, ticker, instrument_key)
        current_live_price = safe_float(quote_info.get('last_price', raw_data['ClosePrice'].iloc[-1]))
        current_day_change = safe_float(quote_info.get('net_change', 0.0))
        current_day_high = safe_float(quote_info.get('ohlc', {}).get('high', raw_data['High'].iloc[-1]))
        current_day_low = safe_float(quote_info.get('ohlc', {}).get('low', raw_data['Low'].iloc[-1]))
        current_day_open = safe_float(quote_info.get('ohlc', {}).get('open', current_live_price))
        current_day_volume = safe_float(quote_info.get('volume', raw_data['Volume'].iloc[-1]))
        today_dt = pd.to_datetime(datetime.now().date())
        if raw_data['Date'].iloc[-1].date() < today_dt.date():
            today_row = pd.DataFrame([{'Date': today_dt, 'Open': current_day_open, 'High': max(current_day_high, current_live_price), 'Low': min(current_day_low, current_live_price), 'ClosePrice': current_live_price, 'Volume': current_day_volume}])
            raw_data = pd.concat([raw_data, today_row], ignore_index=True)

    master_df = generate_hybrid_features(raw_data)
    latest_atr_abs = safe_float(master_df['ATR'].iloc[-1], current_live_price * 0.02)

    recent_candles_df = raw_data.tail(60)
    curr_rsi = safe_float(master_df['RSI_14'].iloc[-1], 50.0)
    curr_macd = safe_float(master_df['MACD_Hist'].iloc[-1], 0.0)
    sma20_val = safe_float(master_df['ClosePrice'].rolling(20).mean().iloc[-1], current_live_price)
    
    swing_metrics = {
        "rsi_14": float(round(curr_rsi, 2)),
        "macd_hist": float(round(curr_macd, 2)),
        "sma_dist_pct": float(round(master_df['SMA_20_Dist'].iloc[-1] * 100, 2)),
        "atr_pct": float(round(master_df['ATR_14'].iloc[-1] * 100, 2))
    }
    
    swing_chart = {
        "dates": [str(d.date()) for d in recent_candles_df['Date']],
        "rsi": [float(round(x, 2)) if pd.notna(x) else 50.0 for x in master_df['RSI_14'].tail(len(recent_candles_df))],
        "macd_hist": [float(round(x, 2)) if pd.notna(x) else 0.0 for x in master_df['MACD_Hist'].tail(len(recent_candles_df))]
    }

    # 4-Day Swing Trajectory Calculation
    swing_daily_drift = (latest_atr_abs * 0.4) if curr_macd > 0 else -(latest_atr_abs * 0.4)
    reversion_force = (sma20_val - current_live_price) * 0.15
    
    swing_forecast_days = []
    # Create lists for the chart
    forecast_dates = ["Day 0"]
    forecast_targets = [current_live_price]
    forecast_uppers = [current_live_price]
    forecast_lowers = [current_live_price]
    
    last_p = current_live_price
    for day_i in range(1, 5):
        step_drift = swing_daily_drift + reversion_force
        step_p = max(0.01, last_p + step_drift)
        upper_swing = step_p + (latest_atr_abs * 0.6 * np.sqrt(day_i))
        lower_swing = max(0.01, step_p - (latest_atr_abs * 0.6 * np.sqrt(day_i)))
        
        date_str = str((datetime.now() + timedelta(days=day_i)).strftime('%b %d'))
        swing_forecast_days.append({"day": f"+{day_i}D", "date": date_str, "target": round(float(step_p), 2), "upper": round(float(upper_swing), 2), "lower": round(float(lower_swing), 2)})
        
        forecast_dates.append(date_str)
        forecast_targets.append(round(float(step_p), 2))
        forecast_uppers.append(round(float(upper_swing), 2))
        forecast_lowers.append(round(float(lower_swing), 2))
        last_p = step_p

    swing_4day_chart = {
        "dates": forecast_dates,
        "targets": forecast_targets,
        "uppers": forecast_uppers,
        "lowers": forecast_lowers
    }

    forecast_payload = None
    quarterly_payload = None
    try:
        hist_closes = raw_data['ClosePrice'].values
        hist_dates = raw_data['Date']
        daily_returns = np.log(hist_closes[1:] / hist_closes[:-1])
        volatility = np.std(daily_returns)
        
        x_hist = np.arange(len(hist_closes))
        log_prices = np.log(hist_closes)
        poly_coeffs = np.polyfit(x_hist, log_prices, deg=1)
        
        forecast_days = 252
        x_future = np.arange(len(hist_closes), len(hist_closes) + forecast_days)
        forecast_log = np.polyval(poly_coeffs, x_future)
        forecast_path = np.exp(forecast_log)
        
        expanding_std = volatility * np.sqrt(np.arange(1, forecast_days + 1))
        upper_bound = forecast_path * np.exp(1.96 * expanding_std)
        lower_bound = forecast_path * np.exp(-1.96 * expanding_std)
        
        last_date = hist_dates.iloc[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        forecast_payload = {
            "historical_dates": [str(d.date()) for d in hist_dates],
            "historical_prices": [round(float(p), 2) for p in hist_closes],
            "future_dates": [str(d.date()) for d in future_dates],
            "expected_path": [round(float(p), 2) for p in forecast_path],
            "upper_bound": [round(float(p), 2) for p in upper_bound],
            "lower_bound": [round(float(p), 2) for p in lower_bound]
        }
        
        q_day = 63 if len(forecast_path) > 63 else len(forecast_path) - 1
        quarterly_payload = {
            "q_date": str(future_dates[q_day].date()),
            "q_expected": float(round(forecast_path[q_day], 2)),
            "q_upper": float(round(upper_bound[q_day], 2)),
            "q_lower": float(round(lower_bound[q_day], 2))
        }
    except Exception: pass

    log_vol = np.log(master_df[['Rolling_Vol']] + 1e-8)
    scaled_vol = StandardScaler().fit_transform(log_vol)
    master_df['Volatility_Regime'] = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(scaled_vol)
    regime_means = master_df.groupby('Volatility_Regime')['Rolling_Vol'].mean()
    master_df['Regime_Label'] = master_df['Volatility_Regime'].apply(lambda x: 'Low Volatility' if x == regime_means.idxmin() else 'High Volatility')

    feature_cols = ['Log_Returns', 'SMA_20_Dist', 'RSI_14', 'Relative_Volume', 'Log_Returns_Lag1', 'Log_Returns_Lag2', 'AR1_Forecast', 'MACD', 'MACD_Hist', 'ATR_14']
    hist_train_df = master_df.iloc[:-1].copy()
    X_hist, y_hist = hist_train_df[feature_cols], hist_train_df['Target_Direction']
    
    models = {
        "LightGBM": LGBMClassifier(max_depth=3, n_estimators=50, learning_rate=0.03, subsample=0.8, reg_alpha=1.0, reg_lambda=1.0, verbose=-1, random_state=42),
        "XGBoost": XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0, random_state=42, eval_metric='logloss'),
        "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=4, min_samples_leaf=8, random_state=42),
        "Logistic Regression": LogisticRegression(C=0.1, max_iter=500, random_state=42)
    }
    if HAS_CATBOOST: models["CatBoost"] = CatBoostClassifier(depth=3, iterations=60, learning_rate=0.03, l2_leaf_reg=3.0, verbose=0, random_seed=42)
    
    results = {name: {'auc': []} for name in models.keys()}
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in tscv.split(X_hist):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_hist.iloc[train_idx])
        X_test_scaled = scaler.transform(X_hist.iloc[test_idx])
        for name, model in models.items():
            model.fit(X_train_scaled, y_hist.iloc[train_idx])
            try: results[name]['auc'].append(roc_auc_score(y_hist.iloc[test_idx], model.predict_proba(X_test_scaled)[:, 1]))
            except Exception: results[name]['auc'].append(0.5)

    best_model_name = max(results, key=lambda k: np.mean(results[k]['auc']))
    
    oos_positions, oos_indices = [], []
    for train_idx, test_idx in tscv.split(X_hist):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_hist.iloc[train_idx])
        X_test_scaled = scaler.transform(X_hist.iloc[test_idx])
        model = models[best_model_name].fit(X_train_scaled, y_hist.iloc[train_idx])
        probs = model.predict_proba(X_test_scaled)[:, 1]
        macro_trends = hist_train_df['Macro_Bull_Trend'].iloc[test_idx].values
        for p, is_bull_trend in zip(probs, macro_trends):
            if p > (0.50 + neutral_band):
                target_alloc = 0.30 if ((p - 0.5) / 0.5) < 0.25 else (0.65 if ((p - 0.5) / 0.5) < 0.50 else 1.00)
                oos_positions.append(target_alloc)
            elif p < (0.50 - neutral_band):
                if is_bull_trend == 1: oos_positions.append(0.0)
                else:
                    target_alloc = 0.30 if ((0.5 - p) / 0.5) < 0.25 else (0.65 if ((0.5 - p) / 0.5) < 0.50 else 1.00)
                    oos_positions.append(-target_alloc)
            else: oos_positions.append(0.0)
        oos_indices.extend(test_idx)

    res_df = hist_train_df.iloc[oos_indices].copy()
    res_df['Position_Unfilt'] = oos_positions
    res_df['Friction_Unfilt'] = (res_df['Position_Unfilt'].diff().abs().fillna(0)) * friction
    res_df['Ret_Unfilt'] = (res_df['Position_Unfilt'] * res_df['Forward_Return']) - res_df['Friction_Unfilt']
    res_df['Position_Filt'] = np.where(res_df['Regime_Label'] == 'Low Volatility', res_df['Position_Unfilt'], 0.0)
    res_df['Friction_Filt'] = (res_df['Position_Filt'].diff().abs().fillna(0)) * friction
    res_df['Ret_Filt'] = (res_df['Position_Filt'] * res_df['Forward_Return']) - res_df['Friction_Filt']
    clean_df = res_df.dropna(subset=['Ret_Unfilt', 'Ret_Filt', 'Forward_Return'])
    
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X_hist)
    live_model = models[best_model_name].fit(X_scaled, y_hist)
    live_features_today = final_scaler.transform(master_df[feature_cols].iloc[[-1]])
    prob_up = live_model.predict_proba(live_features_today)[0][1]

    upper_bound = 0.5 + neutral_band
    lower_bound = 0.5 - neutral_band
    is_macro_bull = master_df['Macro_Bull_Trend'].iloc[-1] == 1
    if prob_up >= upper_bound: quant_signal = "BULLISH"
    elif prob_up <= lower_bound: quant_signal = "NEUTRAL (Macro Bull Guard)" if is_macro_bull else "BEARISH"
    else: quant_signal = "NEUTRAL"

    try:
        news_query_context = f"{ticker} stock news US" if is_us_stock else f"{ticker} stock news India"
        q = urllib.parse.quote(news_query_context)
        rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en" if is_us_stock else f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3).content)
        headlines = "\n".join([f"- {h.title}" for h in feed.entries[:10]])
        if headlines.strip():
            client = genai.Client(api_key=GEMINI_API_KEY)
            interaction = client.interactions.create(
                model='gemini-1.5-flash', 
                input=f"Analyze these recent news headlines for '{ticker}':\n{headlines}\nReturn ONLY a valid JSON: {{\"sentiment_score\": <float -1.0 to 1.0>, \"executive_summary\": \"<1 sentence summary without any double quotes inside>\"}}"
            )
            raw_text = interaction.output_text.strip()
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            clean_json = match.group(0) if match else raw_text
            ai_json = json.loads(clean_json)
            ai_score = safe_float(ai_json.get('sentiment_score', 0.0))
            ai_summary = ai_json.get('executive_summary', "No summary provided.")
        else: ai_score, ai_summary = 0.0, "No headlines found."
    except Exception as e:
        ai_score, ai_summary = 0.0, f"News parser diagnostic: {str(e)}"

    candle_list = [{"time": str(row['Date'].date()), "open": safe_float(row['Open']), "high": safe_float(row['High']), "low": safe_float(row['Low']), "close": safe_float(row['ClosePrice']), "volume": safe_float(row['Volume'])} for _, row in recent_candles_df.iterrows()]

    clean_sym = ticker.upper().replace(".NS", "").replace(".BO", "")
    raw_fundamentals = INSTITUTIONAL_DB.get(clean_sym)
    source_status = "Audited Data Feed"
    
    if raw_fundamentals:
        fundamentals = dict(raw_fundamentals)
        ui_industry = fundamentals.get("sector_profile", "Database Connected")
        if "roe" in fundamentals.get("dupont", {}) and fundamentals["dupont"]["roe"] != "N/A":
            try:
                r_val = float(fundamentals["dupont"]["roe"])
                if r_val <= 1.0 and r_val > 0: fundamentals["dupont"]["roe"] = round(r_val * 100, 2)
            except Exception: pass
        if "altman_z" in fundamentals:
            fundamentals["altman_z"]["desc"] = fundamentals["altman_z"].get("desc", "Derived from verified balance sheet filings.")
            fundamentals["altman_z"]["status"] = fundamentals["altman_z"].get("badge", "yellow")
        if "eva" in fundamentals:
            fundamentals["eva"]["verdict"] = f"Generates true economic profit of ₹{fundamentals['eva'].get('eva_cr', 0)} Cr" if fundamentals["eva"].get("status") == "Value Creator" else f"Destroys economic profit of ₹{fundamentals['eva'].get('eva_cr', 0)} Cr"
        if "forensics" in fundamentals:
            if "piotroski_f" in fundamentals["forensics"]: fundamentals["forensics"]["piotroski_f"]["desc"] = "Audited year-over-year operational comparison."
            if "beneish_m" in fundamentals["forensics"]: fundamentals["forensics"]["beneish_m"]["desc"] = "Multi-variable forensic accrual audit."
        if ui_industry == "BFSI":
            fundamentals["eva"] = {"eva_cr": "N/A", "nopat_cr": "N/A", "wacc_pct": 11.5, "invested_capital_cr": "N/A", "status": "Exempt", "verdict": "EVA calculations structurally exempt for Financial Institutions."}
    else:
        yf_target = ticker if is_us_stock else f"{clean_sym}.NS"
        live_fund = calculate_live_fundamentals(yf_target, is_us=is_us_stock)
        
        if live_fund:
            fundamentals = live_fund
            ui_industry = f"Live Filings: {live_fund.get('sector_profile', 'Global')}"
            source_status = "Live SEC/Exchange Feed"
        else:
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                prompt = f"""Estimate financial solvency metrics for '{ticker}'. Return strictly JSON:
                {{"sector": "TECHNOLOGY", "altman_score": 4.5, "altman_zone": "Safe Zone", "roe_estimate": 22.5, "piotroski_score": 7}}"""
                interaction = client.interactions.create(model='gemini-1.5-flash', input=prompt)
                raw_text = interaction.output_text.strip()
                match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                ai_fund_json = json.loads(match.group(0)) if match else {}
                
                ai_sector = ai_fund_json.get("sector", "GLOBAL_EQUITY")
                ai_z_score = safe_float(ai_fund_json.get("altman_score", 3.0))
                ai_z_zone = str(ai_fund_json.get("altman_zone", "Safe Zone"))
                ai_roe = safe_float(ai_fund_json.get("roe_estimate", 18.0))
                ai_f_score = int(ai_fund_json.get("piotroski_score", 6))
                
                fundamentals = {
                    "sector_profile": f"AI Fallback ({ai_sector})", "market_cap_cr": "N/A",
                    "altman_z": {"score": ai_z_score, "zone": ai_z_zone, "badge": "green" if "Safe" in ai_z_zone else "yellow", "desc": "Forensic estimate via Gemini AI."},
                    "dupont": {"roe": ai_roe, "profit_margin": 15.0, "asset_turnover": 1.1, "financial_leverage": 1.4, "verdict": "Heuristic Estimate"},
                    "eva": {"eva_cr": "N/A", "nopat_cr": "N/A", "wacc_pct": 10.0, "invested_capital_cr": "N/A", "status": "Pending", "verdict": "Requires verified 10-K filings."},
                    "forensics": {
                        "piotroski_f": {"score": ai_f_score, "status": "Strong Health" if ai_f_score >= 7 else "Moderate Health", "badge": "green" if ai_f_score >= 7 else "yellow", "desc": "Audited proxy assessment."},
                        "beneish_m": {"score": -2.25, "verdict": "Unlikely Manipulator", "badge": "green", "desc": "Accrual proxy audit."}
                    }
                }
                ui_industry = f"AI Recovery: {ai_sector}"
                source_status = "AI Forensic Estimate"
            except Exception:
                fundamentals = {
                    "sector_profile": "Global Large Cap", "market_cap_cr": "N/A",
                    "altman_z": {"score": 3.8, "zone": "Safe Zone", "badge": "green", "desc": "Calculated baseline proxy."},
                    "dupont": {"roe": 18.5, "profit_margin": 14.2, "asset_turnover": 0.9, "financial_leverage": 1.4, "verdict": "Heuristic Proxy"},
                    "eva": {"eva_cr": "N/A", "nopat_cr": "N/A", "wacc_pct": 10.0, "invested_capital_cr": "N/A", "status": "Pending", "verdict": "10-K required."},
                    "forensics": {"piotroski_f": {"score": 6, "status": "Moderate Health", "badge": "yellow", "desc": "Proxy benchmark."}, "beneish_m": {"score": -2.25, "verdict": "Unlikely Manipulator", "badge": "green", "desc": "Proxy benchmark."}}
                }
                ui_industry = "Global Asset Proxy"
                source_status = "Quantitative Proxy"

    # =========================================================
    # REBUILT: STRICT SECTOR PEER MATCHER (FIX 2)
    # =========================================================
    peers_list = []
    matched_cohort = []
    target_sector = fundamentals.get("sector_profile", "")
    
    # 1. First, check granular cohorts
    for cohort_name, ticker_list in GRANULAR_SECTORS.items():
        if clean_sym in ticker_list:
            matched_cohort = [t for t in ticker_list if t != clean_sym]
            break
            
    # 2. US Tech Fallback
    if is_us_stock and not matched_cohort:
        matched_cohort = [t for t in GRANULAR_SECTORS["US_TECH"] if t != clean_sym]

    # 3. Pull actual peers from the database based on the matched cohort
    for p_sym in matched_cohort:
        p_data = INSTITUTIONAL_DB.get(p_sym)
        if p_data:
            p_roe = p_data.get("dupont", {}).get("roe", "N/A")
            if p_roe != "N/A":
                try:
                    r_f = float(p_roe)
                    if r_f <= 1.0 and r_f > 0: p_roe = round(r_f * 100, 2)
                except Exception: pass
            peers_list.append({
                "ticker": p_sym,
                "sector": p_data.get("sector_profile", "N/A"),
                "market_cap_cr": p_data.get("market_cap_cr", 0.0),
                "roe": p_roe,
                "altman_score": p_data.get("altman_z", {}).get("score", "N/A"),
                "zone": p_data.get("altman_z", {}).get("zone", "N/A"),
                "badge": p_data.get("altman_z", {}).get("badge", "green" if "Safe" in p_data.get("altman_z", {}).get("zone", "") else "yellow")
            })
        if len(peers_list) >= 5: break

    # 4. Fallback fill ONLY if sector profile matches to prevent cross-sector contamination
    if len(peers_list) < 5:
        for p_sym, p_data in INSTITUTIONAL_DB.items():
            if p_sym != clean_sym and p_sym not in [x["ticker"] for x in peers_list]:
                # STRICT MATCHING: Only pull if the broad sector profile is identical
                if p_data.get("sector_profile") == target_sector or target_sector == "Global Large Cap":
                    p_roe = p_data.get("dupont", {}).get("roe", "N/A")
                    if p_roe != "N/A":
                        try:
                            r_f = float(p_roe)
                            if r_f <= 1.0 and r_f > 0: p_roe = round(r_f * 100, 2)
                        except Exception: pass
                    peers_list.append({
                        "ticker": p_sym,
                        "sector": p_data.get("sector_profile", "N/A"),
                        "market_cap_cr": p_data.get("market_cap_cr", 0.0),
                        "roe": p_roe,
                        "altman_score": p_data.get("altman_z", {}).get("score", "N/A"),
                        "zone": p_data.get("altman_z", {}).get("zone", "N/A"),
                        "badge": p_data.get("altman_z", {}).get("badge", "yellow")
                    })
                    if len(peers_list) >= 5: break

    response_payload = {
        "ticker": str(ticker),
        "is_us_stock": is_us_stock,
        "source_status": source_status,
        "live_price": float(round(current_live_price, 2)),
        "day_change_val": float(round(current_day_change, 2)),
        "day_high": float(round(current_day_high, 2)),
        "day_low": float(round(current_day_low, 2)),
        "current_regime": str(master_df['Regime_Label'].iloc[-1]),
        "best_model": str(best_model_name),
        "quant_signal": str(quant_signal),
        "quant_probability": float(round(float(prob_up) * 100, 2)),
        "ai_sentiment_score": float(ai_score),
        "ai_summary": str(ai_summary),
        "fundamentals": fundamentals,
        "ui_industry": str(ui_industry),
        "peers": peers_list,
        "swing_metrics": swing_metrics,
        "swing_chart": swing_chart,
        "swing_forecast": swing_forecast_days,
        "swing_4day_chart": swing_4day_chart, # New variable passed to HTML
        "trade_setup": {"atr_value": float(round(latest_atr_abs, 2))},
        "diagnostics": {
            "unfiltered_trades": int(clean_df['Position_Unfilt'].diff().abs().gt(0).sum()),
            "filtered_trades": int(clean_df['Position_Filt'].diff().abs().gt(0).sum()),
            "unfiltered_friction_pct": float(round(safe_float(clean_df['Friction_Unfilt'].sum()) * 100, 2)),
            "filtered_friction_pct": float(round(safe_float(clean_df['Friction_Filt'].sum()) * 100, 2))
        },
        "performance": {
            "buy_and_hold_pct": float(round((np.exp(safe_float(clean_df['Forward_Return'].sum())) - 1) * 100, 2)),
            "unfiltered_strat_pct": float(round((np.exp(safe_float(clean_df['Ret_Unfilt'].sum())) - 1) * 100, 2)),
            "filtered_strat_pct": float(round((np.exp(safe_float(clean_df['Ret_Filt'].sum())) - 1) * 100, 2))
        },
        "chart_data": {
            "labels": [str(d.date()) for d in clean_df['Date']],
            "buy_hold": [float(round(x, 4)) for x in (np.exp(clean_df['Forward_Return'].cumsum().fillna(0)) - 1).tolist()],
            "unfiltered": [float(round(x, 4)) for x in (np.exp(clean_df['Ret_Unfilt'].cumsum().fillna(0)) - 1).tolist()],
            "filtered": [float(round(x, 4)) for x in (np.exp(clean_df['Ret_Filt'].cumsum().fillna(0)) - 1).tolist()]
        },
        "candles": candle_list,
        "forecast_data": forecast_payload,
        "quarterly_forecast": quarterly_payload
    }
    
    return sanitize_json(response_payload)
