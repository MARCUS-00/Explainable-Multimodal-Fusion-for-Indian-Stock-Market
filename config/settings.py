"""
config/settings.py
==================
Single source of truth for ALL configuration.
Every module in this project imports from here. Nothing is hardcoded elsewhere.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")

# ══════════════════════════════════════════════════════════════════════════════
# STOCK UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════
STOCKS = [
    "HDFCBANK",  "ICICIBANK",  "SBIN",       "AXISBANK",  "KOTAKBANK",
    "BAJFINANCE","BAJAJFINSV", "INDUSINDBK", "TCS",       "INFY",
    "HCLTECH",   "WIPRO",      "TECHM",      "RELIANCE",  "ONGC",
    "NTPC",      "POWERGRID",  "BPCL",       "HINDUNILVR","ITC",
    "NESTLEIND", "BRITANNIA",  "MARUTI",     "M&M",       "BHARTIARTL",
    "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "SUNPHARMA", "CIPLA",
    "DRREDDY",   "TATASTEEL",  "JSWSTEEL",   "HINDALCO",  "COALINDIA",
    "LT",        "ULTRACEMCO", "GRASIM",     "ASIANPAINT","TITAN",
]

STOCK_COUNT = len(STOCKS)
STOCKS      = STOCKS[:STOCK_COUNT]
STOCKS_NS   = [s + ".NS" for s in STOCKS]

SYMBOL_FALLBACKS = {
    "M&M.NS":  ["M&M.NS", "MM.NS", "M%26M.NS"],
    "ONGC.NS": ["ONGC.NS", "ONGC.BO"],
}

MC_SLUG_MAP = {
    "HDFCBANK":   "hdfc-bank",       "ICICIBANK":  "icici-bank",
    "SBIN":       "state-bank-of-india", "AXISBANK": "axis-bank",
    "KOTAKBANK":  "kotak-mahindra-bank", "BAJFINANCE": "bajaj-finance",
    "BAJAJFINSV": "bajaj-finserv",   "INDUSINDBK": "indusind-bank",
    "TCS":        "tcs",             "INFY":       "infosys",
    "HCLTECH":    "hcl-technologies","WIPRO":      "wipro",
    "TECHM":      "tech-mahindra",   "RELIANCE":   "reliance-industries",
    "ONGC":       "ongc",            "NTPC":       "ntpc",
    "POWERGRID":  "power-grid-corporation-of-india", "BPCL": "bpcl",
    "HINDUNILVR": "hindustan-unilever", "ITC":      "itc",
    "NESTLEIND":  "nestle-india",    "BRITANNIA":  "britannia",
    "MARUTI":     "maruti-suzuki",   "M&M":        "mahindra-and-mahindra",
    "BHARTIARTL": "bharti-airtel",   "EICHERMOT":  "eicher-motors",
    "HEROMOTOCO": "hero-motocorp",   "BAJAJ-AUTO": "bajaj-auto",
    "SUNPHARMA":  "sun-pharmaceutical-industries", "CIPLA": "cipla",
    "DRREDDY":    "dr-reddys-laboratories", "TATASTEEL": "tata-steel",
    "JSWSTEEL":   "jsw-steel",       "HINDALCO":   "hindalco",
    "COALINDIA":  "coal-india",      "LT":         "larsen-and-toubro",
    "ULTRACEMCO": "ultratech-cement","GRASIM":     "grasim",
    "ASIANPAINT": "asian-paints",    "TITAN":      "titan",
}

# ══════════════════════════════════════════════════════════════════════════════
# DATE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DATE_START = "2020-01-01"
DATE_END   = "2026-05-01"

TRAIN_END  = "2024-06-30"
VAL_START  = "2024-07-01"
VAL_END    = "2024-12-31"
TEST_START = "2025-01-01"

# ══════════════════════════════════════════════════════════════════════════════
# FILE PATHS
# ══════════════════════════════════════════════════════════════════════════════
# Data
TECHNICAL_CSV   = os.path.join(DATA_DIR, "technical",   "technical.csv")
FUNDAMENTAL_CSV = os.path.join(DATA_DIR, "fundamental", "fundamental.csv")
NEWS_CSV        = os.path.join(DATA_DIR, "news",        "news.csv")
FINBERT_CSV     = os.path.join(DATA_DIR, "news",        "finbert_scores.csv")
EVENTS_CSV      = os.path.join(DATA_DIR, "events",      "events.csv")
MERGED_CSV      = os.path.join(DATA_DIR, "merged",      "merged_final.csv")
NIFTY_CACHE     = os.path.join(DATA_DIR, "merged",      "_nifty_cache.csv")

# Models  (XGBoost + LSTM only — LightGBM removed, it was never used by ensemble)
MODELS_DIR      = os.path.join(BASE_DIR, "models")
XGB_MODEL_PATH  = os.path.join(MODELS_DIR, "xgboost", "saved", "xgb_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm",    "saved", "lstm_model.pt")

# Tuning artefacts
OPTUNA_BEST_PARAMS   = os.path.join(MODELS_DIR, "xgboost", "saved", "best_params.json")
DECAY_CONSTANTS_PATH = os.path.join(MODELS_DIR, "xgboost", "saved", "decay_constants.json")

# Results
EVAL_DIR              = os.path.join(BASE_DIR, "evaluation")
XGB_RESULTS_PATH      = os.path.join(EVAL_DIR, "results", "xgboost_results.csv")
LSTM_RESULTS_PATH     = os.path.join(EVAL_DIR, "results", "lstm_results.csv")
ENSEMBLE_RESULTS_PATH = os.path.join(EVAL_DIR, "results", "ensemble_results.csv")
WATCHLIST_OUTPUT_PATH = os.path.join(EVAL_DIR, "results", "watchlist_latest.csv")

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════
PREDICTION_THRESHOLD     = 0.50
WATCHLIST_MIN_CONFIDENCE = 0.60

# ══════════════════════════════════════════════════════════════════════════════
# LABEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
LABEL_HORIZON   = 10
LABEL_THRESHOLD = 0.010
RANDOM_SEED     = 42

EXT_TO_INT       = {-1: 0,  1: 1}
INT_TO_EXT       = { 0: -1, 1: 1}
LABEL_MAP_INV    = { 0: "DOWN", 1: "UP"}
DIRECTION_LABELS = ["DOWN", "UP"]

# ══════════════════════════════════════════════════════════════════════════════
# SECTOR MAPPING
# ══════════════════════════════════════════════════════════════════════════════
SECTOR_MAP = {
    "HDFCBANK":   "Financial_Services", "ICICIBANK":  "Financial_Services",
    "SBIN":       "Financial_Services", "AXISBANK":   "Financial_Services",
    "KOTAKBANK":  "Financial_Services", "BAJFINANCE": "Financial_Services",
    "BAJAJFINSV": "Financial_Services", "INDUSINDBK": "Financial_Services",
    "TCS":        "Information_Technology", "INFY":   "Information_Technology",
    "HCLTECH":    "Information_Technology", "WIPRO":  "Information_Technology",
    "TECHM":      "Information_Technology", "RELIANCE": "Energy",
    "ONGC":       "Energy",             "NTPC":       "Utilities",
    "POWERGRID":  "Utilities",          "BPCL":       "Energy",
    "HINDUNILVR": "Consumer_Staples",   "ITC":        "Consumer_Staples",
    "NESTLEIND":  "Consumer_Staples",   "BRITANNIA":  "Consumer_Staples",
    "MARUTI":     "Automobile",         "M&M":        "Automobile",
    "BHARTIARTL": "Telecom",            "EICHERMOT":  "Automobile",
    "HEROMOTOCO": "Automobile",         "BAJAJ-AUTO": "Automobile",
    "SUNPHARMA":  "Pharma",             "CIPLA":      "Pharma",
    "DRREDDY":    "Pharma",             "TATASTEEL":  "Metals",
    "JSWSTEEL":   "Metals",             "HINDALCO":   "Metals",
    "COALINDIA":  "Energy",             "LT":         "Industrials",
    "ULTRACEMCO": "Cement",             "GRASIM":     "Cement",
    "ASIANPAINT": "Consumer_Durables",  "TITAN":      "Consumer_Durables",
}
SECTOR_NAMES   = sorted(set(SECTOR_MAP.values()))
SECTOR_TO_CODE = {name: i for i, name in enumerate(SECTOR_NAMES)}

# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
XGBOOST_PARAMS = {
    "n_estimators":     200,
    "max_depth":        3,
    "learning_rate":    0.04,
    "subsample":        0.75,
    "colsample_bytree": 0.65,
    "min_child_weight": 8,
    "gamma":            0.2,
    "reg_alpha":        0.4,
    "reg_lambda":       1.5,
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
    "tree_method":      "hist",
}

XGBOOST_FEATURES = [
    "ret_lag_1d", "ret_lag_3d", "ret_lag_5d",
    "price_to_ema20", "price_to_ema50", "ema_cross",
    "rsi_norm", "macd_hist_norm",
    "momentum_5d", "momentum_10d", "momentum_diff", "momentum_strength",
    "atr_ratio", "bb_pct",
    "volatility_5d", "volatility_10d", "hist_vol_20d",
    "obv_change", "vol_spike", "vol_breakout", "price_pos_20d",
    "nifty_ret_5d", "ret_vs_nifty_1d", "ret_vs_nifty_5d",
    "sector_ret_1d", "sector_ret_5d", "sector_ret_loo",
    "sector_rel_momentum", "sector_encoded",
    "PE_Ratio", "ROE", "Revenue_Growth", "Profit_Growth",
    "has_fundamental", "days_since_fundamental",
    "news_score", "news_pos", "news_neg",
    "news_decay", "news_sentiment_mom",
    "news_count_1d", "news_count_3d",
    "days_since_news", "has_news",
    "is_rbi", "is_gdp", "is_cpi", "is_budget", "is_pmi",
    "is_earnings", "is_dividend",
    "mom_z_5", "mom_z_10", "vol_z",
    "pct_from_52h", "cs_rank_ret", "cs_rank_mom", "consec_dir",
    "market_return_20d", "nifty_trend_20d", "rolling_volatility_20d",
    "relative_strength_vs_market", "bullish_regime", "bearish_regime",
    "bb_rsi_combo", "vol_adj_mom", "rsi_x_mom", "mom_vol_ratio",
]

# ══════════════════════════════════════════════════════════════════════════════
# OPTUNA / LSTM / FINBERT / DECAY / REGIME CONSTANTS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
OPTUNA_TRIALS       = 40
OPTUNA_QUICK_TRIALS = 10

SEQUENCE_LENGTH = 15
LSTM_HIDDEN     = 64
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.3

FINBERT_MODEL      = "ProsusAI/finbert"
FINBERT_MAX_LEN    = 128
FINBERT_BATCH_SIZE = 32

DECAY_K_CANDIDATES = [1, 2, 3, 5, 7, 10, 14]
DEFAULT_DECAY_K    = 5

REGIME_BULL_ADJ   = 0.05
REGIME_BEAR_ADJ   = 0.08
REGIME_MIN_THRESH = 0.52
REGIME_MAX_THRESH = 0.75

# ══════════════════════════════════════════════════════════════════════════════
# ENSURE OUTPUT DIRECTORIES EXIST
# ══════════════════════════════════════════════════════════════════════════════
for _d in [
    DATA_DIR,
    os.path.join(DATA_DIR, "technical"),
    os.path.join(DATA_DIR, "fundamental"),
    os.path.join(DATA_DIR, "news"),
    os.path.join(DATA_DIR, "events"),
    os.path.join(DATA_DIR, "merged"),
    os.path.join(MODELS_DIR, "xgboost", "saved"),
    os.path.join(MODELS_DIR, "lstm",    "saved"),
    os.path.join(EVAL_DIR,   "results"),
]:
    os.makedirs(_d, exist_ok=True)
