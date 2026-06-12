# ════════════════════════════════════════════════════════════════════
# BIST 100 VaR Risk Analytics Platform — Production (Render Ready)
# BAL620 · Canbolat Karaduman
# Düzeltmeler:
#   1. Analiz "Çalıştır" butonuyla tetikleniyor (gereksiz yeniden hesaplama yok)
#   2. In-memory veri cache aktif
#   3. VaR formülü düzeltildi: kayıp = W*(z*σ*√T - μ*T)
#   4. CVaR formülü düzeltildi: W*σ*√T*φ(z)/(1-α) - W*μ*T
#   5. Sharpe risk-free rate günlük bazda doğru uygulanıyor
#   6. Beta hesabı BIST100 endeksiyle fetch_data içinde yapılıyor
#   7. GARCH tarih hizalaması düzeltildi
#   8. Modal ctx.states hatası düzeltildi
#   9. Render deploy: host=0.0.0.0, server expose, gunicorn uyumlu
# ════════════════════════════════════════════════════════════════════

import dash
from dash import dcc, html, Input, Output, State, dash_table, ALL, ctx, no_update
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
import warnings
import json

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════
# 1. VERİ SETİ
# ════════════════════════════════════════════════════════
BIST100 = {
    'AKBNK': ('Akbank', 'Banka'), 'GARAN': ('Garanti BBVA', 'Banka'),
    'HALKB': ('Halkbank', 'Banka'), 'ISCTR': ('İş Bankası (C)', 'Banka'),
    'VAKBN': ('Vakıfbank', 'Banka'), 'YKBNK': ('Yapı Kredi Bankası', 'Banka'),
    'TSKB': ('TSKB', 'Banka'), 'ALBRK': ('Albaraka Türk', 'Banka'),
    'QNBFB': ('QNB Finansbank', 'Banka'), 'KLNMA': ('Kalkınma Yatırım Bankası', 'Banka'),
    'KCHOL': ('Koç Holding', 'Holding'), 'SAHOL': ('Sabancı Holding', 'Holding'),
    'DOHOL': ('Doğan Holding', 'Holding'), 'GLYHO': ('Global Yatırım Holding', 'Holding'),
    'ISMEN': ('İş Yatırım', 'Aracı Kurum'),
    'EREGL': ('Ereğli Demir Çelik', 'Sanayi'), 'KRDMD': ('Kardemir (D)', 'Sanayi'),
    'ARCLK': ('Arçelik', 'Sanayi'), 'VESTL': ('Vestel Elektronik', 'Sanayi'),
    'VESBE': ('Vestel Beyaz Eşya', 'Sanayi'), 'BRSAN': ('Borusan Mannesmann', 'Sanayi'),
    'SISE': ('Şişe Cam', 'Sanayi'), 'TRKCM': ('Trakya Cam', 'Sanayi'),
    'ISDMR': ('İskenderun Demir Çelik', 'Sanayi'), 'ADEL': ('Adel Kalemcilik', 'Sanayi'),
    'ERBOS': ('Erbosan', 'Sanayi'), 'SARKY': ('Sarkuysan', 'Sanayi'),
    'TTRAK': ('Türk Traktör', 'Otomotiv'), 'TOASO': ('Tofaş', 'Otomotiv'),
    'FROTO': ('Ford Otosan', 'Otomotiv'), 'DOAS': ('Doğuş Otomotiv', 'Otomotiv'),
    'ASUZU': ('Anadolu Isuzu', 'Otomotiv'), 'KARSN': ('Karsan Otomotiv', 'Otomotiv'),
    'TUPRS': ('Tüpraş', 'Enerji'), 'AYGAZ': ('Aygaz', 'Enerji'),
    'AKSEN': ('Aksa Enerji', 'Enerji'), 'ZOREN': ('Zorlu Enerji', 'Enerji'),
    'ODAS': ('Odaş Elektrik', 'Enerji'),
    'ENKAI': ('Enka İnşaat', 'İnşaat'), 'AKCNS': ('Akçansa Çimento', 'İnşaat'),
    'CIMSA': ('Çimsa Çimento', 'İnşaat'), 'TKFEN': ('Tekfen Holding', 'İnşaat'),
    'OYAKC': ('Oyak Çimento', 'İnşaat'), 'GOLTS': ('Göltaş Çimento', 'İnşaat'),
    'THYAO': ('Türk Hava Yolları', 'Havacılık'), 'PGSUS': ('Pegasus Havayolları', 'Havacılık'),
    'TAVHL': ('TAV Havalimanları', 'Havacılık'), 'CLEBI': ('Çelebi Hava Servisi', 'Havacılık'),
    'BIMAS': ('BİM Mağazalar', 'Perakende'), 'SOKM': ('ŞOK Marketler', 'Perakende'),
    'MAVI': ('Mavi Giyim', 'Perakende'), 'MGROS': ('Migros Ticaret', 'Perakende'),
    'ASELS': ('Aselsan', 'Savunma'),
    'LOGO': ('Logo Yazılım', 'Teknoloji'), 'INDES': ('İndeks Bilgisayar', 'Teknoloji'),
    'KONTR': ('Kontrolmatik', 'Teknoloji'), 'KAREL': ('Karel Elektronik', 'Teknoloji'),
    'NETAS': ('Netaş Telekomünikasyon', 'Teknoloji'),
    'TCELL': ('Turkcell', 'Telekom'), 'TTKOM': ('Türk Telekom', 'Telekom'),
    'ANSGR': ('Anadolu Sigorta', 'Sigorta'), 'AKGRT': ('Aksigorta', 'Sigorta'),
    'RAYSG': ('Ray Sigorta', 'Sigorta'),
    'ISGYO': ('İş GYO', 'GYO'), 'EKGYO': ('Emlak Konut GYO', 'GYO'),
    'ECILC': ('Eczacıbaşı İlaç', 'Kimya'), 'PETKM': ('Petkim Petrokimya', 'Kimya'),
    'GUBRF': ('Gübre Fabrikaları', 'Kimya'), 'DEVA': ('Deva Holding', 'Kimya'),
    'SELEC': ('Selçuk Ecza', 'Kimya'),
    'AEFES': ('Anadolu Efes', 'Gıda'), 'CCOLA': ('Coca-Cola İçecek', 'Gıda'),
    'TATGD': ('Tat Gıda', 'Gıda'), 'BANVT': ('Banvit', 'Gıda'),
    'ULKER': ('Ülker Bisküvi', 'Gıda'),
    'MPARK': ('MLP Sağlık', 'Sağlık'),
    'PRKME': ('Park Elektrik', 'Madencilik'), 'KRDMA': ('Kardemir (A)', 'Madencilik'),
}

FALLBACK = {
    'AKBNK': {'mu': 0.00052, 'sigma': 0.0215, 'beta': 1.02, 'price': 67.85, 'change': -0.32},
    'GARAN': {'mu': 0.00054, 'sigma': 0.0225, 'beta': 1.08, 'price': 89.40, 'change': 1.23},
    'HALKB': {'mu': 0.00031, 'sigma': 0.0268, 'beta': 1.15, 'price': 28.90, 'change': -1.24},
    'ISCTR': {'mu': 0.00044, 'sigma': 0.0198, 'beta': 0.98, 'price': 56.20, 'change': 0.54},
    'VAKBN': {'mu': 0.00041, 'sigma': 0.0245, 'beta': 1.09, 'price': 33.25, 'change': 0.90},
    'YKBNK': {'mu': 0.00043, 'sigma': 0.0232, 'beta': 1.05, 'price': 43.10, 'change': 1.78},
    'TSKB':  {'mu': 0.00038, 'sigma': 0.0210, 'beta': 0.92, 'price': 12.85, 'change': 0.47},
    'ALBRK': {'mu': 0.00028, 'sigma': 0.0255, 'beta': 1.10, 'price': 8.42,  'change': -0.60},
    'QNBFB': {'mu': 0.00035, 'sigma': 0.0222, 'beta': 1.01, 'price': 14.18, 'change': 0.21},
    'KLNMA': {'mu': 0.00035, 'sigma': 0.0245, 'beta': 1.05, 'price': 18.42, 'change': -0.11},
    'KCHOL': {'mu': 0.00052, 'sigma': 0.0178, 'beta': 0.85, 'price': 198.40, 'change': 0.82},
    'SAHOL': {'mu': 0.00050, 'sigma': 0.0185, 'beta': 0.88, 'price': 67.20, 'change': 0.36},
    'DOHOL': {'mu': 0.00042, 'sigma': 0.0205, 'beta': 0.95, 'price': 24.60, 'change': 1.10},
    'GLYHO': {'mu': 0.00038, 'sigma': 0.0248, 'beta': 1.08, 'price': 34.20, 'change': -0.85},
    'ISMEN': {'mu': 0.00040, 'sigma': 0.0230, 'beta': 1.00, 'price': 28.60, 'change': 0.42},
    'EREGL': {'mu': 0.00050, 'sigma': 0.0210, 'beta': 0.92, 'price': 46.54, 'change': -0.45},
    'KRDMD': {'mu': 0.00038, 'sigma': 0.0248, 'beta': 1.08, 'price': 18.72, 'change': 0.64},
    'ARCLK': {'mu': 0.00048, 'sigma': 0.0215, 'beta': 0.96, 'price': 148.40, 'change': -0.31},
    'VESTL': {'mu': 0.00042, 'sigma': 0.0252, 'beta': 1.10, 'price': 38.90, 'change': 0.77},
    'VESBE': {'mu': 0.00045, 'sigma': 0.0238, 'beta': 1.05, 'price': 74.30, 'change': -1.02},
    'BRSAN': {'mu': 0.00042, 'sigma': 0.0225, 'beta': 0.95, 'price': 88.50, 'change': -0.44},
    'SISE':  {'mu': 0.00049, 'sigma': 0.0200, 'beta': 0.90, 'price': 62.40, 'change': -0.18},
    'TRKCM': {'mu': 0.00046, 'sigma': 0.0208, 'beta': 0.92, 'price': 58.70, 'change': 0.65},
    'ISDMR': {'mu': 0.00040, 'sigma': 0.0248, 'beta': 1.08, 'price': 24.80, 'change': -0.44},
    'ADEL':  {'mu': 0.00048, 'sigma': 0.0228, 'beta': 0.98, 'price': 84.60, 'change': 0.46},
    'ERBOS': {'mu': 0.00044, 'sigma': 0.0238, 'beta': 1.02, 'price': 48.20, 'change': 0.42},
    'SARKY': {'mu': 0.00046, 'sigma': 0.0232, 'beta': 1.00, 'price': 74.40, 'change': -0.27},
    'TTRAK': {'mu': 0.00060, 'sigma': 0.0195, 'beta': 0.87, 'price': 428.60, 'change': 1.20},
    'TOASO': {'mu': 0.00044, 'sigma': 0.0215, 'beta': 0.96, 'price': 156.20, 'change': -0.54},
    'FROTO': {'mu': 0.00062, 'sigma': 0.0225, 'beta': 1.01, 'price': 942.50, 'change': 1.12},
    'DOAS':  {'mu': 0.00050, 'sigma': 0.0220, 'beta': 0.97, 'price': 148.80, 'change': 0.83},
    'ASUZU': {'mu': 0.00052, 'sigma': 0.0235, 'beta': 1.03, 'price': 164.80, 'change': 0.92},
    'KARSN': {'mu': 0.00042, 'sigma': 0.0248, 'beta': 1.08, 'price': 44.20,  'change': 1.42},
    'TUPRS': {'mu': 0.00062, 'sigma': 0.0195, 'beta': 0.87, 'price': 180.20, 'change': 0.67},
    'AYGAZ': {'mu': 0.00040, 'sigma': 0.0180, 'beta': 0.78, 'price': 92.60,  'change': 0.33},
    'AKSEN': {'mu': 0.00048, 'sigma': 0.0228, 'beta': 1.02, 'price': 43.80,  'change': 1.54},
    'ZOREN': {'mu': 0.00040, 'sigma': 0.0252, 'beta': 1.12, 'price': 18.94,  'change': -0.73},
    'ODAS':  {'mu': 0.00038, 'sigma': 0.0265, 'beta': 1.18, 'price': 31.25,  'change': 2.38},
    'ENKAI': {'mu': 0.00044, 'sigma': 0.0188, 'beta': 0.82, 'price': 32.76,  'change': 0.12},
    'AKCNS': {'mu': 0.00048, 'sigma': 0.0205, 'beta': 0.90, 'price': 128.40, 'change': 0.54},
    'CIMSA': {'mu': 0.00045, 'sigma': 0.0210, 'beta': 0.92, 'price': 94.60,  'change': 0.88},
    'TKFEN': {'mu': 0.00050, 'sigma': 0.0215, 'beta': 0.95, 'price': 98.40,  'change': -0.28},
    'OYAKC': {'mu': 0.00045, 'sigma': 0.0205, 'beta': 0.90, 'price': 54.80,  'change': 0.28},
    'GOLTS': {'mu': 0.00038, 'sigma': 0.0230, 'beta': 1.00, 'price': 32.60,  'change': 0.56},
    'THYAO': {'mu': 0.00082, 'sigma': 0.0285, 'beta': 1.32, 'price': 298.60, 'change': 2.14},
    'PGSUS': {'mu': 0.00065, 'sigma': 0.0312, 'beta': 1.18, 'price': 521.40, 'change': -0.85},
    'TAVHL': {'mu': 0.00058, 'sigma': 0.0258, 'beta': 1.14, 'price': 248.60, 'change': 1.64},
    'CLEBI': {'mu': 0.00055, 'sigma': 0.0268, 'beta': 1.18, 'price': 183.20, 'change': 1.22},
    'BIMAS': {'mu': 0.00042, 'sigma': 0.0168, 'beta': 0.72, 'price': 389.40, 'change': 0.22},
    'SOKM':  {'mu': 0.00040, 'sigma': 0.0222, 'beta': 0.94, 'price': 78.30,  'change': -0.68},
    'MAVI':  {'mu': 0.00058, 'sigma': 0.0245, 'beta': 1.06, 'price': 184.60, 'change': 1.35},
    'MGROS': {'mu': 0.00048, 'sigma': 0.0205, 'beta': 0.89, 'price': 234.60, 'change': 1.45},
    'ASELS': {'mu': 0.00065, 'sigma': 0.0248, 'beta': 1.12, 'price': 83.45,  'change': 3.21},
    'LOGO':  {'mu': 0.00072, 'sigma': 0.0295, 'beta': 1.22, 'price': 142.80, 'change': -1.87},
    'INDES': {'mu': 0.00060, 'sigma': 0.0278, 'beta': 1.20, 'price': 286.40, 'change': 1.14},
    'KONTR': {'mu': 0.00065, 'sigma': 0.0288, 'beta': 1.24, 'price': 74.60,  'change': 1.87},
    'KAREL': {'mu': 0.00055, 'sigma': 0.0305, 'beta': 1.30, 'price': 98.60,  'change': -1.44},
    'NETAS': {'mu': 0.00052, 'sigma': 0.0315, 'beta': 1.28, 'price': 94.60,  'change': 0.95},
    'TCELL': {'mu': 0.00048, 'sigma': 0.0185, 'beta': 0.80, 'price': 82.40,  'change': 0.58},
    'TTKOM': {'mu': 0.00040, 'sigma': 0.0175, 'beta': 0.75, 'price': 42.20,  'change': 0.14},
    'ANSGR': {'mu': 0.00042, 'sigma': 0.0215, 'beta': 0.90, 'price': 52.40,  'change': 0.48},
    'AKGRT': {'mu': 0.00040, 'sigma': 0.0205, 'beta': 0.88, 'price': 38.60,  'change': -0.26},
    'RAYSG': {'mu': 0.00038, 'sigma': 0.0228, 'beta': 0.98, 'price': 14.52,  'change': 0.69},
    'ISGYO': {'mu': 0.00035, 'sigma': 0.0232, 'beta': 0.98, 'price': 14.38,  'change': 0.84},
    'EKGYO': {'mu': 0.00038, 'sigma': 0.0242, 'beta': 1.04, 'price': 12.64,  'change': 1.52},
    'ECILC': {'mu': 0.00048, 'sigma': 0.0218, 'beta': 0.94, 'price': 64.80,  'change': 0.72},
    'PETKM': {'mu': 0.00048, 'sigma': 0.0222, 'beta': 0.98, 'price': 34.60,  'change': 0.74},
    'GUBRF': {'mu': 0.00040, 'sigma': 0.0230, 'beta': 0.98, 'price': 54.20,  'change': 1.45},
    'DEVA':  {'mu': 0.00044, 'sigma': 0.0235, 'beta': 1.02, 'price': 38.20,  'change': -0.52},
    'SELEC': {'mu': 0.00040, 'sigma': 0.0225, 'beta': 0.96, 'price': 52.60,  'change': 0.34},
    'AEFES': {'mu': 0.00050, 'sigma': 0.0192, 'beta': 0.85, 'price': 128.60, 'change': 0.44},
    'CCOLA': {'mu': 0.00055, 'sigma': 0.0185, 'beta': 0.80, 'price': 194.80, 'change': 0.92},
    'TATGD': {'mu': 0.00042, 'sigma': 0.0215, 'beta': 0.93, 'price': 42.80,  'change': -0.37},
    'BANVT': {'mu': 0.00038, 'sigma': 0.0235, 'beta': 1.02, 'price': 62.40,  'change': 1.18},
    'ULKER': {'mu': 0.00044, 'sigma': 0.0198, 'beta': 0.85, 'price': 92.30,  'change': 0.22},
    'MPARK': {'mu': 0.00055, 'sigma': 0.0225, 'beta': 0.98, 'price': 102.40, 'change': 2.15},
    'PRKME': {'mu': 0.00045, 'sigma': 0.0248, 'beta': 1.08, 'price': 34.60,  'change': 0.73},
    'KRDMA': {'mu': 0.00036, 'sigma': 0.0252, 'beta': 1.10, 'price': 16.80,  'change': 0.60},
}

SEKTORLER = sorted(set(v[1] for v in BIST100.values()))
COLORS = ['#FF9900', '#2F74D0', '#2A9D8F', '#E63946', '#b794f4',
          '#76e4f7', '#fbb6ce', '#9ae6b4', '#fed7aa', '#c3dafe']

# ════════════════════════════════════════════════════════
# 2. VERİ MOTORU  (cache aktif)
# ════════════════════════════════════════════════════════
_data_cache: dict = {}

def fetch_data(ticker: str, period: str = '1y') -> dict:
    """yfinance'dan veri çek; başarısız olursa FALLBACK kullan. Sonucu cache'le."""
    key = f"{ticker}_{period}"
    if key in _data_cache:
        return _data_cache[key]
    try:
        df = yf.Ticker(f'{ticker}.IS').history(period=period, auto_adjust=True)
        if df.empty or len(df) < 20:
            raise ValueError("Yetersiz veri")
        df['ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)
        mu    = float(df['ret'].mean())
        sigma = float(df['ret'].std())
        price = float(df['Close'].iloc[-1])
        change = float((df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100)
        # Beta hesabı: BIST100 endeksiyle regresyon
        try:
            bist_df = yf.Ticker('XU100.IS').history(period=period, auto_adjust=True)
            bist_df['ret'] = np.log(bist_df['Close'] / bist_df['Close'].shift(1))
            bist_df.dropna(inplace=True)
            common = df.index.intersection(bist_df.index)
            if len(common) > 30:
                cov_mat = np.cov(df.loc[common, 'ret'], bist_df.loc[common, 'ret'])
                beta = float(cov_mat[0, 1] / cov_mat[1, 1])
            else:
                beta = FALLBACK.get(ticker, {}).get('beta', 1.0)
        except Exception:
            beta = FALLBACK.get(ticker, {}).get('beta', 1.0)

        result = {
            'closes':  df['Close'].round(2).tolist(),
            'returns': df['ret'].tolist(),
            'dates':   [str(d.date()) for d in df.index],
            'mu':      mu,
            'sigma':   sigma,
            'beta':    round(beta, 3),
            'price':   round(price, 2),
            'change':  round(change, 2),
            'kaynak':  'Yahoo Finance ✓',
        }
        _data_cache[key] = result
        return result
    except Exception:
        fb = FALLBACK.get(ticker, {'mu': 0.0005, 'sigma': 0.022, 'beta': 1.0, 'price': 50.0, 'change': 0.0})
        np.random.seed(abs(hash(ticker)) % (2**32))
        mock_rets   = np.random.normal(fb['mu'], fb['sigma'], 252).tolist()
        mock_prices = (fb['price'] * np.cumprod(1 + np.array(mock_rets))).tolist()
        mock_dates  = [str(d.date()) for d in pd.date_range(end=pd.Timestamp.now(), periods=252)]
        result = {
            'closes':  mock_prices, 'returns': mock_rets, 'dates': mock_dates,
            'mu': fb['mu'], 'sigma': fb['sigma'], 'beta': fb.get('beta', 1.0),
            'price': fb['price'], 'change': fb['change'], 'kaynak': 'Fallback ⚠',
        }
        _data_cache[key] = result
        return result


def fit_garch(returns: list, horizon: int = 1) -> dict:
    """GARCH(1,1) modeli. Hata durumunda {'error': mesaj} döner."""
    try:
        from arch import arch_model
        r = pd.Series(returns).dropna() * 100
        if r.std() < 1e-6:
            return {'error': 'Varyans çok düşük'}
        res = arch_model(r, vol='Garch', p=1, q=1, dist='normal',
                         rescale=False).fit(disp='off', show_warning=False)
        fc = res.forecast(horizon=horizon, reindex=False)
        return {
            'cond_vol': (res.conditional_volatility / 100).tolist(),
            'omega':    float(res.params.get('omega', 0)),
            'alpha':    float(res.params.get('alpha[1]', 0)),
            'beta_g':   float(res.params.get('beta[1]', 0)),
            'tahmin':   float(np.sqrt(fc.variance.values[-1, :].mean())) / 100,
            'error':    None,
        }
    except Exception as e:
        return {'error': str(e)[:120]}


# ════════════════════════════════════════════════════════
# 3. RİSK MOTORİ  (düzeltilmiş formüller)
# ════════════════════════════════════════════════════════
# Formüller (JP Morgan RiskMetrics standardı):
#
#   z            = norm.ppf(α)          α = güven seviyesi (ör. 0.99)
#                  z pozitif (sağ kuyruk quantile'ı; kayıp solda)
#
#   Parametrik VaR = W × (z × σ × √T  −  μ × T)
#   (pozitif → kayıp tutarı; T gün ileriye projeksiyon)
#
#   CVaR (Expected Shortfall):
#     CVaR = W × √T × σ × φ(z) / (1−α)  −  W × μ × T
#   (VaR'ın ötesindeki koşullu ortalama kayıp)
#
#   Sharpe = (μ_yıllık − r_f) / σ_yıllık
#     r_f = 0.40  (TCMB politika faizine göre yaklaşık yıllık oran, ondalık)
#     μ_yıllık = μ_günlük × 252
#     σ_yıllık = σ_günlük × √252
#
#   VaR Katkısı_i = w_i × W × (z × σ_i − μ_i)   (bileşen VaR yaklaşımı)

RF_ANNUAL = 0.40   # Türkiye risk-free (TCMB yaklaşımı, yıllık ondalık)
RF_DAILY  = RF_ANNUAL / 252

def run_analysis(tickers: list, weights_dict: dict,
                 pv: float, ci: float, hz: int, mc_n: int, period: str) -> dict:
    h = tickers
    w_raw = np.array([weights_dict.get(k, 100.0 / len(h)) for k in h])
    w = w_raw / w_raw.sum()   # normalize → toplamı 1

    # Veri çek
    meta = {}
    for k in h:
        d = fetch_data(k, period)
        d['sirket'] = BIST100.get(k, (k, '-'))[0]
        d['sektor']  = BIST100.get(k, ('-', '-'))[1]
        meta[k] = d

    mu_v    = np.array([meta[k]['mu']    for k in h])
    sig_v   = np.array([meta[k]['sigma'] for k in h])
    beta_v  = np.array([meta[k].get('beta', 1.0) for k in h])
    rets_l  = [meta[k]['returns'] for k in h]

    # Korelasyon matrisi
    if all(len(r) > 30 for r in rets_l):
        ml  = min(len(r) for r in rets_l)
        dff = pd.DataFrame({k: meta[k]['returns'][-ml:] for k in h})
        corr    = dff.corr()
        p_rets  = dff.dot(w).tolist()   # portföy getiri serisi (GARCH için)
    else:
        corr   = pd.DataFrame(np.eye(len(h)), index=h, columns=h)
        p_rets = (np.random.default_rng(42)
                    .normal(float(w @ mu_v), float(w @ sig_v), 252)).tolist()

    cov   = np.outer(sig_v, sig_v) * corr.values
    p_mu  = float(w @ mu_v)
    p_sig = float(np.sqrt(w @ cov @ w))
    p_beta= float(w @ beta_v)

    # ── Parametrik VaR (düzeltilmiş) ──────────────────────────────
    # z = norm.ppf(α) > 0  (örn. %99 → z ≈ 2.326)
    z     = float(norm.ppf(ci))          # pozitif
    sqrtT = float(np.sqrt(hz))
    # Kayıp = W × (z×σ×√T − μ×T)  →  negatif getiri senaryosu
    d_var = pv * (z * p_sig * sqrtT - p_mu * hz)
    d_var = max(d_var, 0.0)              # teorik minimum 0

    vp    = d_var / pv * 100             # portföy değerine oran %

    # ── CVaR (düzeltilmiş) ────────────────────────────────────────
    # CVaR = W × √T × σ × φ(z)/(1−α) − W × μ × T
    phi_z = float(norm.pdf(z))
    cvar  = pv * sqrtT * p_sig * phi_z / (1 - ci) - pv * p_mu * hz
    cvar  = max(cvar, d_var)             # CVaR ≥ VaR her zaman

    # ── Sharpe (düzeltilmiş) ──────────────────────────────────────
    mu_annual  = p_mu  * 252
    sig_annual = p_sig * np.sqrt(252)
    sharpe = (mu_annual - RF_ANNUAL) / sig_annual if sig_annual > 0 else 0.0

    # ── Monte Carlo ───────────────────────────────────────────────
    rng  = np.random.default_rng(42)
    pl   = pv * (p_mu * hz + p_sig * sqrtT * rng.standard_normal(mc_n))
    pl_s = np.sort(pl)
    idx_var  = max(int((1 - ci) * mc_n) - 1, 0)
    mc_var   = abs(float(pl_s[idx_var]))
    mc_worst = abs(float(pl_s[0]))
    mc_med   = float(np.median(pl_s))

    # ── Senaryo analizi ───────────────────────────────────────────
    sen = {
        'boga': float(pv * (p_mu + 2 * p_sig) * sqrtT),
        'baz':  float(pv * p_mu * hz),
        'ayi':  float(pv * (p_mu - 2 * p_sig) * sqrtT),
        'kriz': float(pv * (p_mu - 3.5 * p_sig) * sqrtT),
    }

    # ── GARCH ─────────────────────────────────────────────────────
    garch = fit_garch(p_rets, horizon=hz)

    # ── Katkı tablosu (Component VaR yaklaşımı) ───────────────────
    katki_rows = []
    for i, k in enumerate(h):
        # Bileşen VaR_i = w_i × W × (z × σ_i − μ_i)
        comp_var = pv * w[i] * (z * meta[k]['sigma'] - meta[k]['mu'])
        katki_rows.append({
            'Kod':        k,
            'Şirket':     meta[k]['sirket'],
            'Sektör':     meta[k]['sektor'],
            'Fiyat ₺':    round(meta[k]['price'], 2),
            'Değişim %':  round(meta[k]['change'], 2),
            'Ağırlık %':  round(w[i] * 100, 1),
            'μ/gün %':    round(meta[k]['mu']    * 100, 4),
            'σ/gün %':    round(meta[k]['sigma'] * 100, 4),
            'Beta':       round(float(beta_v[i]), 3),
            'VaR Katkı ₺': round(max(comp_var, 0), 0),
            'Yıllık Vol %': round(meta[k]['sigma'] * np.sqrt(252) * 100, 1),
            'Kaynak':     meta[k]['kaynak'],
        })

    return {
        'p_mu': p_mu, 'p_sig': p_sig, 'p_beta': p_beta,
        'z': z, 'var': d_var, 'vp': vp, 'cvar': cvar, 'sharpe': sharpe,
        'mu_annual': mu_annual * 100, 'sig_annual': sig_annual * 100,
        'mc_var': mc_var, 'mc_worst': mc_worst, 'mc_med': mc_med,
        'pl_s': pl_s.tolist(), 'sen': sen, 'garch': garch,
        'katki': katki_rows, 'corr': corr, 'meta': meta, 'w': w.tolist(),
    }


# ════════════════════════════════════════════════════════
# 4. CSS
# ════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root {
  --bg:#000;--panel:#0d0d0d;--card:#141414;--border:#262626;
  --amber:#FF9900;--blue:#2F74D0;--red:#E63946;--green:#2A9D8F;
  --text:#e0e0e0;--muted:#808080;--font:"IBM Plex Mono",monospace;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:12px;}
.header{background:var(--panel);border-bottom:2px solid var(--amber);
  padding:10px 20px;display:flex;justify-content:space-between;align-items:center;
  position:sticky;top:0;z-index:100;}
.header-title{color:var(--amber);font-weight:600;font-size:15px;letter-spacing:1px;}
.main-grid{display:grid;grid-template-columns:290px 1fr;gap:15px;
  padding:15px;max-width:1600px;margin:0 auto;}
.panel{background:var(--card);border:1px solid var(--border);padding:15px;margin-bottom:15px;}
.panel-title{color:var(--muted);font-size:10px;text-transform:uppercase;
  margin-bottom:12px;border-bottom:1px solid var(--border);padding-bottom:5px;letter-spacing:.8px;}
.search-box{width:100%;background:var(--bg);border:1px solid var(--border);
  color:var(--amber);padding:8px;font-family:var(--font);margin-bottom:8px;
  outline:none;font-size:11px;}
.search-box:focus{border-color:var(--amber);}
.stock-item{display:flex;justify-content:space-between;align-items:center;
  padding:6px 8px;cursor:pointer;border-bottom:1px solid #1a1a1a;transition:background .12s;}
.stock-item:hover{background:#1a1a1a;}
.stock-item.selected{border-left:3px solid var(--amber);background:#111;color:var(--amber);}
.stock-code{font-weight:600;font-size:12px;}
.stock-name{font-size:9px;color:var(--muted);}
.param-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}
.run-btn{width:100%;padding:11px;background:var(--amber);color:#000;border:none;
  font-family:var(--font);font-size:13px;font-weight:700;cursor:pointer;
  letter-spacing:.06em;transition:background .2s;margin-top:8px;}
.run-btn:hover:not([disabled]){background:#cc7a00;}
.run-btn[disabled]{opacity:.4;cursor:not-allowed;}
.btn-outline{background:transparent;color:var(--amber);border:1px solid var(--amber);
  padding:6px 10px;cursor:pointer;font-family:var(--font);font-size:10px;width:100%;
  transition:background .15s;}
.btn-outline:hover{background:rgba(255,153,0,.1);}
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:15px;}
.metric-card{background:var(--panel);border:1px solid var(--border);padding:14px;
  border-top:2px solid var(--blue);}
.metric-card.red-top{border-top-color:var(--red);}
.metric-card.amber-top{border-top-color:var(--amber);}
.metric-card.green-top{border-top-color:var(--green);}
.metric-label{color:var(--muted);font-size:10px;margin-bottom:4px;}
.metric-val{font-size:20px;font-weight:700;margin-top:4px;}
.metric-sub{font-size:9px;margin-top:4px;color:var(--muted);}
.val-red{color:var(--red);}.val-green{color:var(--green);}
.val-amber{color:var(--amber);}.val-blue{color:var(--blue);}
.custom-tabs{border-bottom:1px solid var(--border);margin-bottom:15px;display:flex;}
.tab{background:transparent!important;border:none!important;
  border-bottom:2px solid transparent!important;color:var(--muted)!important;
  font-family:var(--font)!important;font-size:10px!important;
  text-transform:uppercase!important;padding:10px 12px!important;
  cursor:pointer;letter-spacing:.5px;}
.tab-active{color:var(--amber)!important;border-bottom:2px solid var(--amber)!important;
  font-weight:600;}
.alert-box{background:#070f1a;border:1px solid #1a365d;color:#5096ed;
  padding:9px 14px;margin-bottom:12px;font-size:11px;}
.alert-danger{background:#1a0707;border:1px solid #5c1a1a;color:var(--red);
  padding:9px 14px;margin-bottom:12px;font-size:11px;}
.alert-warn{background:#1a1507;border:1px solid #5c4a1a;color:var(--amber);
  padding:9px 14px;margin-bottom:12px;font-size:11px;}
.scenario-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}
.scenario-card{background:var(--panel);border:1px solid var(--border);padding:12px;}
.scenario-title{font-size:10px;color:var(--muted);text-transform:uppercase;margin-bottom:5px;}
.scenario-val{font-size:18px;font-weight:700;}
.formula-box{margin-top:14px;padding:12px;background:var(--panel);
  border:1px solid var(--border);font-size:10px;color:var(--muted);line-height:1.9;}
.modal-overlay{position:fixed;top:0;left:0;width:100%;height:100%;
  background:rgba(0,0,0,.8);z-index:1000;display:none;
  align-items:center;justify-content:center;}
.modal-overlay.open{display:flex;}
.modal-content{background:var(--card);border:1px solid var(--amber);
  padding:25px;box-shadow:0 0 20px rgba(255,153,0,.2);width:480px;max-height:85vh;
  overflow-y:auto;}
.rc-slider-track{background-color:var(--amber)!important;}
.rc-slider-handle{border-color:var(--amber)!important;background:var(--amber)!important;
  width:14px!important;height:14px!important;margin-top:-5px!important;}
.rc-slider-rail{background:#1a1a1a!important;}
.rc-slider-mark-text{color:#fff!important;font-family:var(--font)!important;font-size:10px!important;}
.chips{display:flex;flex-wrap:wrap;gap:5px;min-height:20px;margin:8px 0;}
.chip{display:inline-flex;align-items:center;gap:4px;padding:3px 8px;
  background:rgba(255,153,0,.1);border:1px solid rgba(255,153,0,.3);
  font-size:11px;color:var(--amber);}
.chip-x{cursor:pointer;color:var(--muted);}
.chip-x:hover{color:var(--red);}
@media(max-width:1100px){
  .main-grid{grid-template-columns:1fr;}
  .metric-grid{grid-template-columns:repeat(2,1fr);}
  .scenario-grid{grid-template-columns:repeat(2,1fr);}
}
"""

PL = dict(
    paper_bgcolor='#141414', plot_bgcolor='#141414',
    font=dict(family='IBM Plex Mono', color='#e0e0e0', size=11),
    margin=dict(l=55, r=20, t=40, b=40),
    xaxis=dict(gridcolor='#262626', zerolinecolor='#262626'),
    yaxis=dict(gridcolor='#262626', zerolinecolor='#262626'),
    legend=dict(bgcolor='#141414', bordercolor='#262626', borderwidth=1),
)

def ftl(v):
    return f"₺{abs(v):,.0f}"


# ════════════════════════════════════════════════════════
# 5. DASH LAYOUT
# ════════════════════════════════════════════════════════
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title='PORT | BIST 100 Risk Analytics',
)
server = app.server   # Gunicorn / Render için

app.index_string = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>PORT | BIST 100 Risk Analytics</title>
    <style>{CSS}</style>
  </head>
  <body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
  </body>
</html>"""

app.layout = html.Div([
    # ── HEADER ──────────────────────────────────────────
    html.Div([
        html.Div("BAL620 PORT · BIST 100 RISK ANALYTICS", className='header-title'),
        html.Div([
            html.Span("⚠ VERİLER 15 DK GECİKMELİ",
                      style={'color': 'var(--amber)', 'marginRight': '15px', 'fontWeight': '600'}),
            html.Span("RENDER · PRODUCTION",
                      style={'color': 'var(--green)'}),
        ]),
    ], className='header'),

    html.Div([
        # ── SOL PANEL ───────────────────────────────────
        html.Div([
            # 1. Hisse Seçimi
            html.Div("1. HİSSE SEÇİMİ", className='panel-title'),
            dcc.Dropdown(
                id='sector-filter',
                options=([{'label': 'Tümü', 'value': 'Tümü'}] +
                         [{'label': s, 'value': s} for s in SEKTORLER]),
                value='Tümü', clearable=False,
                style={'color': '#000', 'marginBottom': '8px',
                       'fontSize': '11px', 'fontFamily': 'IBM Plex Mono'},
            ),
            dcc.Input(id='srch', type='text',
                      placeholder='Ara: THYAO, GARAN…',
                      className='search-box', debounce=True),
            html.Div(id='stock-list',
                     style={'maxHeight': '200px', 'overflowY': 'auto',
                            'marginBottom': '10px'}),
            # Seçili hisseler chip
            html.Div(id='chips-area', className='chips'),

            # 2. Ağırlık Ayarı
            html.Div("2. PORTFÖY TUTARLARI", className='panel-title',
                     style={'marginTop': '15px'}),
            html.Button('⚖ Tutar & Ağırlık Ayarla',
                        id='btn-open-modal', className='btn-outline'),

            # Toplam portföy değeri
            html.Div([
                html.Span('Portföy Değeri:', style={'color': 'var(--muted)'}),
                html.Span('0 ₺', id='val-pv',
                           style={'color': 'var(--amber)', 'fontWeight': 'bold',
                                  'fontSize': '14px'}),
            ], className='param-row',
               style={'marginTop': '10px', 'borderTop': '1px solid var(--border)',
                      'paddingTop': '10px'}),

            # 3. Risk Parametreleri
            html.Div("3. RİSK PARAMETRELERİ", className='panel-title',
                     style={'marginTop': '15px'}),
            html.Div([html.Span('Güven Seviyesi'),
                      html.Span('%99', id='val-ci',
                                style={'color': '#fff', 'fontWeight': 'bold'})],
                     className='param-row'),
            dcc.Slider(id='ci', min=0.90, max=0.99, step=0.01, value=0.99,
                       marks={0.90: {'label': '%90', 'style': {'color': '#fff', 'fontSize': '10px'}},
                              0.95: {'label': '%95', 'style': {'color': '#fff', 'fontSize': '10px'}},
                              0.99: {'label': '%99', 'style': {'color': '#fff', 'fontSize': '10px'}}}),
            html.Div([html.Span('Zaman Ufku'),
                      html.Span('10 Gün', id='val-hz',
                                style={'color': '#fff', 'fontWeight': 'bold'})],
                     className='param-row', style={'marginTop': '12px'}),
            dcc.Slider(id='hz', min=1, max=252, step=1, value=10,
                       marks={1: {'label': '1', 'style': {'color': '#fff', 'fontSize': '10px'}},
                              63: {'label': '3A', 'style': {'color': '#fff', 'fontSize': '10px'}},
                              126: {'label': '6A', 'style': {'color': '#fff', 'fontSize': '10px'}},
                              252: {'label': '1Y', 'style': {'color': '#fff', 'fontSize': '10px'}}}),
            html.Div('Veri Dönemi', className='param-row',
                     style={'color': 'var(--muted)', 'marginTop': '12px'}),
            dcc.Dropdown(
                id='donem',
                options=[{'label': '6 Ay', 'value': '6mo'},
                         {'label': '1 Yıl', 'value': '1y'},
                         {'label': '2 Yıl', 'value': '2y'}],
                value='1y', clearable=False,
                style={'color': '#000', 'fontSize': '11px', 'fontFamily': 'IBM Plex Mono'},
            ),

            # 4. Monte Carlo
            html.Div("4. MONTE CARLO", className='panel-title', style={'marginTop': '15px'}),
            html.Div([html.Span('Simülasyon'),
                      html.Span('10,000', id='val-mc',
                                style={'color': '#fff', 'fontWeight': 'bold'})],
                     className='param-row'),
            dcc.Slider(id='mc', min=1000, max=50000, step=1000, value=10000,
                       marks={1000:  {'label': '1k',  'style': {'color': '#fff', 'fontSize': '10px'}},
                              10000: {'label': '10k', 'style': {'color': '#fff', 'fontSize': '10px'}},
                              50000: {'label': '50k', 'style': {'color': '#fff', 'fontSize': '10px'}}}),

            # ANALİZ BUTONU
            html.Button('▶ ANALİZ BAŞLAT',
                        id='run-btn', className='run-btn',
                        n_clicks=0, disabled=True),
        ], className='panel'),

        # ── SAĞ İÇERİK ──────────────────────────────────
        html.Div(id='dashboard-content',
                 children=[html.Div(
                     "Sol panelden hisse seçin, tutarları girin, ardından "
                     "ANALİZ BAŞLAT'a tıklayın.",
                     style={'padding': '30px', 'color': 'var(--amber)',
                            'fontFamily': 'IBM Plex Mono', 'textAlign': 'center'})]),
    ], className='main-grid'),

    # ── MODAL ───────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div("PORTFÖY BİLEŞEN TUTARLARI", className='panel-title',
                     style={'color': 'var(--amber)'}),
            html.Div(id='modal-sliders',
                     style={'maxHeight': '380px', 'overflowY': 'auto',
                            'paddingRight': '10px'}),
            html.Div(id='modal-total',
                     style={'color': '#fff', 'fontWeight': 'bold', 'fontSize': '14px',
                            'marginTop': '12px', 'borderTop': '1px solid var(--border)',
                            'paddingTop': '12px', 'textAlign': 'right'}),
            html.Button('KAYDET VE KAPAT',
                        id='btn-close-modal', className='run-btn', n_clicks=0),
        ], className='modal-content'),
    ], id='weight-modal', className='modal-overlay'),

    # ── STORE ────────────────────────────────────────────
    dcc.Store(id='store-selected', data=[]),
    dcc.Store(id='store-amounts',  data={}),
    dcc.Store(id='modal-open',     data=False),
])


# ════════════════════════════════════════════════════════
# 6. CALLBACK'LER
# ════════════════════════════════════════════════════════

# Slider etiketleri
@app.callback(Output('val-ci', 'children'), Input('ci', 'value'))
def lbl_ci(v): return f'%{int(v * 100)}' if v else '—'

@app.callback(Output('val-hz', 'children'), Input('hz', 'value'))
def lbl_hz(v): return f'{v} Gün' if v else '—'

@app.callback(Output('val-mc', 'children'), Input('mc', 'value'))
def lbl_mc(v): return f'{v:,.0f}' if v else '—'


# Hisse listesi
@app.callback(
    Output('stock-list', 'children'),
    Input('srch', 'value'),
    Input('sector-filter', 'value'),
    Input('store-selected', 'data'),
)
def update_list(q, sector, selected):
    q = (q or '').upper()
    selected = selected or []
    items = []
    for ticker, info in BIST100.items():
        if sector != 'Tümü' and info[1] != sector:
            continue
        if q and q not in ticker and q not in info[0].upper():
            continue
        is_sel = ticker in selected
        fb = FALLBACK.get(ticker, {})
        chg = fb.get('change', 0)
        chg_color = 'var(--green)' if chg >= 0 else 'var(--red)'
        items.append(html.Div([
            html.Div([
                html.Div(ticker, className='stock-code'),
                html.Div(info[0], className='stock-name'),
            ]),
            html.Div([
                # Sayıyı yazdıran formülü sildik, yerine sadece '' (boşluk) koyduk
                html.Div('',
                         style={'color': chg_color, 'fontSize': '10px'}),
                
                html.Div('✓' if is_sel else '',
                         style={'color': 'var(--amber)', 'fontWeight': 'bold',
                                'marginLeft': '6px'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ], className=f"stock-item {'selected' if is_sel else ''}",
           id={'type': 'stock-click', 'index': ticker},
           n_clicks=0))
    return items


# Hisse toggle (seç/kaldır)
@app.callback(
    Output('store-selected', 'data'),
    Output('store-amounts', 'data', allow_duplicate=True),
    Input({'type': 'stock-click', 'index': ALL}, 'n_clicks'),
    State({'type': 'stock-click', 'index': ALL}, 'id'),
    State('store-selected', 'data'),
    State('store-amounts', 'data'),
    prevent_initial_call=True,
)
def toggle_stock(clicks, ids, selected, amounts):
    if not ctx.triggered or not any(c for c in (clicks or [])):
        return no_update, no_update
    ticker = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index']
    selected = list(selected or [])
    amounts  = dict(amounts  or {})
    if ticker in selected:
        selected.remove(ticker)
        amounts.pop(ticker, None)
    elif len(selected) < 15:
        selected.append(ticker)
        amounts[ticker] = 0
    return selected, amounts


# Chip'ler
@app.callback(
    Output('chips-area', 'children'),
    Output('run-btn', 'disabled'),
    Input('store-selected', 'data'),
    Input('store-amounts', 'data'),
)
def update_chips(selected, amounts):
    selected = selected or []
    amounts  = amounts  or {}
    chips = [
        html.Div([
            ticker,
            html.Span('✕', className='chip-x',
                      id={'type': 'chip-rm', 'index': ticker}, n_clicks=0),
        ], className='chip') for ticker in selected
    ]
    total = sum(amounts.values())
    disabled = len(selected) < 1 or total <= 0
    return chips, disabled


# Chip'ten sil
@app.callback(
    Output('store-selected', 'data', allow_duplicate=True),
    Input({'type': 'chip-rm', 'index': ALL}, 'n_clicks'),
    State({'type': 'chip-rm', 'index': ALL}, 'id'),
    State('store-selected', 'data'),
    prevent_initial_call=True,
)
def remove_chip(clicks, ids, selected):
    if not ctx.triggered or not any(c for c in (clicks or [])):
        return no_update
    ticker = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index']
    return [x for x in (selected or []) if x != ticker]


# Modal aç/kapat + slider içeriği
@app.callback(
    Output('weight-modal', 'className'),
    Output('modal-sliders', 'children'),
    Input('btn-open-modal', 'n_clicks'),
    Input('btn-close-modal', 'n_clicks'),
    State('store-selected', 'data'),
    State('store-amounts', 'data'),
    prevent_initial_call=True,
)
def handle_modal(open_n, close_n, selected, amounts):
    selected = selected or []
    amounts  = amounts  or {}
    trigger  = ctx.triggered_id
    is_open  = (trigger == 'btn-open-modal')

    rows = []
    for t in selected:
        rows.append(html.Div([
            html.Div(t, style={'color': '#fff', 'fontWeight': 'bold',
                               'width': '75px', 'fontSize': '13px'}),
            dcc.Input(
                type='number',
                id={'type': 'amt-input', 'index': t},
                value=amounts.get(t, 0),
                min=0, step=1000,
                placeholder='0',
                className='search-box',
                style={'width': '170px', 'marginBottom': '0',
                       'marginRight': '12px', 'textAlign': 'right',
                       'fontSize': '13px'},
            ),
            html.Div(id={'type': 'w-pct', 'index': t},
                     style={'color': 'var(--amber)', 'fontWeight': 'bold',
                            'width': '55px', 'textAlign': 'right'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}))

    return ('modal-overlay open' if is_open else 'modal-overlay'), rows


# Tutar inputları → store + yüzde güncelle
@app.callback(
    Output('store-amounts', 'data', allow_duplicate=True),
    Output({'type': 'w-pct', 'index': ALL}, 'children'),
    Input({'type': 'amt-input', 'index': ALL}, 'value'),
    State({'type': 'amt-input', 'index': ALL}, 'id'),
    State('store-amounts', 'data'),
    prevent_initial_call=True,
)
def sync_amounts(vals, ids, amounts):
    amounts = dict(amounts or {})
    for i, val in enumerate(vals):
        amounts[ids[i]['index']] = val or 0
    total = sum(amounts.values())
    pcts = []
    for val in vals:
        v = val or 0
        pcts.append(f'%{v / total * 100:.1f}' if total > 0 else '%0.0')
    return amounts, pcts


# Toplam portföy değeri etiketi
@app.callback(
    Output('val-pv', 'children'),
    Output('modal-total', 'children'),
    Input('store-amounts', 'data'),
)
def update_pv(amounts):
    total = sum((amounts or {}).values())
    return f'{total:,.0f} ₺', f'Toplam: {total:,.0f} ₺'


# ── ANA ANALİZ CALLBACK (sadece butonla tetiklenir) ──────────────
@app.callback(
    Output('dashboard-content', 'children'),
    Input('run-btn', 'n_clicks'),
    State('store-selected', 'data'),
    State('store-amounts', 'data'),
    State('ci', 'value'),
    State('hz', 'value'),
    State('mc', 'value'),
    State('donem', 'value'),
    prevent_initial_call=True,
)
def run_dashboard(n_clicks, selected, amounts, ci, hz, mc_n, donem):
    selected = selected or []
    amounts  = amounts  or {}
    total_pv = sum(amounts.values())

    if not selected:
        return html.Div("En az 1 hisse seçin.",
                        style={'padding': '20px', 'color': 'var(--amber)'})
    if total_pv <= 0:
        return html.Div("Hisse tutarlarını giriniz (Tutar & Ağırlık Ayarla).",
                        style={'color': 'var(--red)', 'padding': '20px'})

    weights = {k: (v / total_pv * 100) for k, v in amounts.items() if k in selected}

    # Risk hesaplama
    R = run_analysis(selected, weights, total_pv, ci, hz, mc_n, donem)

    mu, sig, z = R['p_mu'], R['p_sig'], R['z']
    vp = R['vp']
    vp_label = '🔴 YÜKSEK' if vp > 5 else ('🟡 ORTA' if vp > 2 else '🟢 DÜŞÜK')

    # ── Metrik kartlar ────────────────────────────────────────
    def krt(label, val, sub, cls=''):
        return html.Div([
            html.Div(label, className='metric-label'),
            html.Div(val,   className=f'metric-val {cls}'),
            html.Div(sub,   className='metric-sub'),
        ], className=f'metric-card {cls.replace("val-","")}-top')

    summary = html.Div([
        krt(f'Parametrik VaR (%{ci*100:.0f} · {hz}G)',
            ftl(R['var']),
            f'Portföy Riski: %{vp:.2f} | {vp_label}', 'val-red'),
        krt('Monte Carlo VaR',
            ftl(R['mc_var']),
            f'{mc_n:,} simülasyon', 'val-amber'),
        krt('CVaR / Beklenen Kayıp',
            ftl(R['cvar']),
            'VaR aşımı ort. kaybı', 'val-red'),
        krt('Sharpe Oranı',
            f"{R['sharpe']:.3f}",
            f"Yıllık: μ=%{R['mu_annual']:.1f} σ=%{R['sig_annual']:.1f}", 'val-blue'),
    ], className='metric-grid')

    # ── Grafik yardımcısı ─────────────────────────────────────
    def G(fig, h=320):
        return dcc.Graph(
            figure=fig,
            config={'displayModeBar': True, 'displaylogo': False},
            style={'height': f'{h}px'},
        )

    # ── T1: Dağılım ───────────────────────────────────────────
    x_all = np.linspace(mu - 4.2 * sig, mu + 4.2 * sig, 400)
    vx    = mu - z * sig          # VaR eşiği getiri ekseninde (sol kuyruk)
    f1 = go.Figure()
    f1.add_trace(go.Scatter(
        x=x_all[x_all <= vx] * 100,
        y=norm.pdf(x_all[x_all <= vx], mu, sig) / 100,
        fill='tozeroy', fillcolor='rgba(230,57,70,.35)',
        line=dict(color='rgba(0,0,0,0)'), name='Kayıp Kuyruğu'))
    f1.add_trace(go.Scatter(
        x=x_all[x_all > vx] * 100,
        y=norm.pdf(x_all[x_all > vx], mu, sig) / 100,
        fill='tozeroy', fillcolor='rgba(47,116,208,.15)',
        line=dict(color='rgba(0,0,0,0)'), name='Güvenli Bölge'))
    f1.add_trace(go.Scatter(
        x=x_all * 100, y=norm.pdf(x_all, mu, sig) / 100,
        line=dict(color='#FF9900', width=2), showlegend=False))
    f1.add_vline(
        x=vx * 100,
        line=dict(color='#E63946', width=2, dash='dash'),
        annotation_text=f'VaR %{vp:.2f}',
        annotation_font_color='#E63946')
    f1.update_layout(**PL,
        title=f'Parametrik Getiri Dağılımı   μ={mu*100:.4f}%   σ={sig*100:.4f}%',
        xaxis_title='Günlük Getiri (%)', yaxis_title='Olasılık Yoğunluğu', height=320)

    # ── T2: Monte Carlo ───────────────────────────────────────
    pl_arr = np.array(R['pl_s']) / 1000
    f2 = go.Figure()
    f2.add_trace(go.Histogram(
        x=pl_arr[pl_arr < -R['mc_var'] / 1000],
        nbinsx=25, marker_color='#E63946', opacity=0.75, name='Kayıp'))
    f2.add_trace(go.Histogram(
        x=pl_arr[pl_arr >= -R['mc_var'] / 1000],
        nbinsx=35, marker_color='#2F74D0', opacity=0.6, name='Kâr'))
    f2.add_vline(
        x=-R['mc_var'] / 1000,
        line=dict(color='#FF9900', width=2, dash='dash'),
        annotation_text=f"MC VaR {ftl(R['mc_var'])}",
        annotation_font_color='#FF9900')
    f2.update_layout(**PL, barmode='overlay',
        title=f'Monte Carlo P&L — {mc_n:,} Simülasyon',
        xaxis_title='Kâr/Zarar (Bin ₺)', yaxis_title='Simülasyon Frekansı', height=320)

    # ── T3: Portföy pasta + volatilite çubuğu ─────────────────
    f3a = go.Figure(go.Pie(
        labels=selected,
        values=[round(weights.get(k, 0), 1) for k in selected],
        hole=0.55,
        marker=dict(colors=COLORS[:len(selected)],
                    line=dict(color='#000', width=2))))
    f3a.update_layout(**PL, title='Portföy Ağırlık Dağılımı',
        annotations=[dict(text='Portföy', x=0.5, y=0.5,
                          font_size=13, showarrow=False,
                          font_color='#e0e0e0')], height=300)

    sig_v = np.array([R['meta'][k]['sigma'] for k in selected])
    vols  = sig_v * np.sqrt(252) * 100
    f3b = go.Figure(go.Bar(
        x=selected, y=vols,
        marker_color=['#E63946' if v > 40 else '#FF9900' if v > 25 else '#2A9D8F'
                      for v in vols],
        text=[f'%{v:.1f}' for v in vols], textposition='outside'))
    f3b.update_layout(**PL, showlegend=False,
        title='Yıllık Volatilite (σ × √252)', yaxis_title='%', height=300)

    t3_content = html.Div([
        html.Div([
            html.Div(G(f3a, 300), style={'flex': '1'}),
            html.Div(G(f3b, 300), style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '12px'}),
    ])

    # ── T4: Normalize fiyat ───────────────────────────────────
    f4 = go.Figure()
    for i, k in enumerate(selected[:8]):
        closes = R['meta'][k].get('closes', [])
        if closes:
            arr = np.array(closes) / closes[0] * 100
            f4.add_trace(go.Scatter(
                y=arr.tolist(), name=k,
                line=dict(color=COLORS[i % len(COLORS)], width=1.8)))
    f4.add_hline(y=100, line=dict(color='#808080', width=1, dash='dot'))
    f4.update_layout(**PL,
        title=f'Normalize Fiyat Trendi (Baz=100) — {donem}',
        xaxis_title='İşlem Günü', yaxis_title='Endeks', height=320)

    # ── T5: Korelasyon ────────────────────────────────────────
    cv = R['corr'].values
    f5 = go.Figure(go.Heatmap(
        z=cv, x=selected, y=selected,
        colorscale='RdYlGn', zmin=-1, zmax=1,
        text=[[f'{cv[i][j]:.2f}' for j in range(len(selected))]
              for i in range(len(selected))],
        texttemplate='%{text}', textfont={'size': 9},
        colorbar=dict(title='ρ', tickfont=dict(color='#e0e0e0'))))
    f5.update_layout(**PL, title='Getiri Korelasyon Matrisi', height=320)

    # ── T6: GARCH ─────────────────────────────────────────────
    gh = R['garch']
    f6 = go.Figure()
    if gh.get('error') is None and gh.get('cond_vol'):
        dates_0  = R['meta'][selected[0]]['dates']
        cond_pct = [v * 100 for v in gh['cond_vol']]
        ml       = min(len(dates_0), len(cond_pct))
        f6.add_trace(go.Scatter(
            x=dates_0[-ml:], y=cond_pct[-ml:],
            line=dict(color='#2A9D8F', width=1.5),
            fill='tozeroy', fillcolor='rgba(42,157,143,.1)',
            name='Koşullu Vol'))
        garch_info = (f"ω={gh.get('omega',0):.6f}  "
                      f"α={gh.get('alpha',0):.4f}  "
                      f"β={gh.get('beta_g',0):.4f}  "
                      f"Tahmin: %{gh.get('tahmin',0)*100:.3f}")
        f6.update_layout(**PL,
            title='GARCH(1,1) Portföy Koşullu Volatilitesi',
            xaxis_title='Tarih', yaxis_title='Günlük Vol (%)', height=320)
    else:
        garch_info = f"GARCH: {gh.get('error', 'veri bekleniyor')}"
        f6.add_annotation(text=garch_info, x=0.5, y=0.5, showarrow=False,
                          font=dict(color='#E63946', size=12))
        f6.update_layout(**PL, title='GARCH(1,1) (Hata)', height=320)

    # ── T7: Senaryo ───────────────────────────────────────────
    s = R['sen']
    t7_content = html.Div([
        html.Div(f'{hz} Günlük Stres Testi Senaryoları', className='alert-warn'),
        html.Div([
            html.Div([
                html.Div('🐂 Boğa (+2σ)', className='scenario-title'),
                html.Div(f"+{ftl(s['boga'])}", className='scenario-val',
                         style={'color': 'var(--green)'}),
                html.Div('Beklenen kâr üst sınırı', className='metric-sub'),
            ], className='scenario-card'),
            html.Div([
                html.Div('📊 Baz (μ)', className='scenario-title'),
                html.Div(ftl(abs(s['baz'])) if s['baz'] >= 0 else f"-{ftl(abs(s['baz']))}",
                         className='scenario-val',
                         style={'color': 'var(--blue)'}),
                html.Div('Tarihsel ortalama projeksiyon', className='metric-sub'),
            ], className='scenario-card'),
            html.Div([
                html.Div('🐻 Ayı (−2σ)', className='scenario-title'),
                html.Div(f"-{ftl(abs(s['ayi']))}", className='scenario-val',
                         style={'color': 'var(--amber)'}),
                html.Div('Normal stres kaybı', className='metric-sub'),
            ], className='scenario-card'),
            html.Div([
                html.Div('💥 Kriz (−3.5σ)', className='scenario-title'),
                html.Div(f"-{ftl(abs(s['kriz']))}", className='scenario-val',
                         style={'color': 'var(--red)'}),
                html.Div('Sistemik kriz senaryosu', className='metric-sub'),
            ], className='scenario-card'),
        ], className='scenario-grid'),
    ], style={'padding': '10px'})

    # ── T8: Detay tablosu ─────────────────────────────────────
    rows = R['katki']
    cols = [{'name': c, 'id': c} for c in rows[0].keys()]
    t8_content = html.Div([
        dash_table.DataTable(
            data=rows, columns=cols,
            style_table={'overflowX': 'auto'},
            style_cell={'backgroundColor': '#141414', 'color': '#e0e0e0',
                        'border': '1px solid #262626', 'fontFamily': 'IBM Plex Mono',
                        'fontSize': '10px', 'padding': '8px', 'textAlign': 'center'},
            style_header={'backgroundColor': '#0d0d0d', 'color': '#808080',
                          'fontWeight': '600', 'textTransform': 'uppercase'},
            style_data_conditional=[
                {'if': {'filter_query': '{Değişim %} > 0', 'column_id': 'Değişim %'},
                 'color': '#2A9D8F', 'fontWeight': 'bold'},
                {'if': {'filter_query': '{Değişim %} < 0', 'column_id': 'Değişim %'},
                 'color': '#E63946', 'fontWeight': 'bold'},
                {'if': {'column_id': 'Ağırlık %'}, 'color': '#2F74D0', 'fontWeight': 'bold'},
                {'if': {'column_id': 'VaR Katkı ₺'}, 'color': '#FF9900', 'fontWeight': 'bold'},
                {'if': {'column_id': 'Kod'}, 'color': '#fff', 'fontWeight': 'bold'},
                {'if': {'filter_query': '{Kaynak} contains "Yahoo"', 'column_id': 'Kaynak'},
                 'color': '#2A9D8F'},
                {'if': {'filter_query': '{Kaynak} contains "Fallback"', 'column_id': 'Kaynak'},
                 'color': '#FF9900'},
            ],
            page_size=15,
        ),
        html.Div(
            'VaR = W×(z×σ×√T − μ×T)  |  CVaR = W×√T×σ×φ(z)/(1−α) − W×μ×T  |  '
            f'σₚ² = Σ wᵢwⱼσᵢσⱼρᵢⱼ  |  Sharpe = (μ×252 − rᶠ)/(σ×√252)  |  '
            f'α={ci*100:.0f}%  z={z:.4f}  rᶠ={RF_ANNUAL*100:.0f}%',
            className='formula-box',
        ),
    ], style={'padding': '5px'})

    # ── Sekmeler ──────────────────────────────────────────────
    tabs = dcc.Tabs(
        id='sub-tabs', value='t1', className='custom-tabs',
        children=[
            dcc.Tab(label='📊 DAĞILIM', value='t1',
                    className='tab', selected_className='tab-active',
                    children=[
                        html.Div(
                            f"z={z:.4f}  μ={mu*100:.4f}%  σ={sig*100:.4f}%  "
                            f"VaR = W×(z×σ×√T − μ×T)",
                            className='alert-box'),
                        G(f1),
                    ]),
            dcc.Tab(label='🎲 MONTE CARLO', value='t2',
                    className='tab', selected_className='tab-active',
                    children=[
                        html.Div(
                            f"En Kötü: {ftl(R['mc_worst'])}  "
                            f"Medyan: {ftl(R['mc_med'])}  "
                            f"MC VaR: {ftl(R['mc_var'])}",
                            className='alert-box'),
                        G(f2),
                    ]),
            dcc.Tab(label='💼 PORTFÖY', value='t3',
                    className='tab', selected_className='tab-active',
                    children=[t3_content]),
            dcc.Tab(label='📈 FİYAT', value='t4',
                    className='tab', selected_className='tab-active',
                    children=[G(f4)]),
            dcc.Tab(label='🔗 KORELASYON', value='t5',
                    className='tab', selected_className='tab-active',
                    children=[
                        html.Div('Gerçek getiri serisinden hesaplanan korelasyon matrisi',
                                 className='alert-box'),
                        G(f5),
                    ]),
            dcc.Tab(label='📉 GARCH', value='t6',
                    className='tab', selected_className='tab-active',
                    children=[
                        html.Div(garch_info, className='alert-box'),
                        G(f6),
                    ]),
            dcc.Tab(label='🔮 SENARYO', value='t7',
                    className='tab', selected_className='tab-active',
                    children=[t7_content]),
            dcc.Tab(label='📋 DETAY', value='t8',
                    className='tab', selected_className='tab-active',
                    children=[t8_content]),
        ],
    )

    return html.Div([
        summary,
        (html.Div('⚠ Portföy riski kritik eşiğin üzerinde (%5+).',
                  className='alert-danger') if vp > 5 else html.Div()),
        html.Div(tabs, className='panel', style={'padding': '12px'}),
    ])


# ════════════════════════════════════════════════════════
# 7. ENTRYPOINT
# ════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
