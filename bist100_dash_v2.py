import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
import warnings, json
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, dash_table, ALL

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════
# BIST 100 — TAM LİSTE
# ════════════════════════════════════════════════════════
BIST100 = {
    'AKBNK':('Akbank','Banka'),
    'GARAN':('Garanti BBVA','Banka'),
    'HALKB':('Halkbank','Banka'),
    'ISCTR':('İş Bankası (C)','Banka'),
    'VAKBN':('Vakıfbank','Banka'),
    'YKBNK':('Yapı Kredi Bankası','Banka'),
    'TSKB':('TSKB','Banka'),
    'ALBRK':('Albaraka Türk','Banka'),
    'QNBFB':('QNB Finansbank','Banka'),
    'KLNMA':('Kalkınma Yatırım Bankası','Banka'),
    'KCHOL':('Koç Holding','Holding'),
    'SAHOL':('Sabancı Holding','Holding'),
    'DOHOL':('Doğan Holding','Holding'),
    'MGROS':('Migros Ticaret','Holding'),
    'MPARK':('MLP Sağlık','Holding'),
    'GLYHO':('Global Yatırım Holding','Holding'),
    'ISMEN':('İş Yatırım','Holding'),
    'EREGL':('Ereğli Demir Çelik','Sanayi'),
    'KRDMD':('Kardemir (D)','Sanayi'),
    'TTRAK':('Türk Traktör','Sanayi'),
    'ARCLK':('Arçelik','Sanayi'),
    'VESTL':('Vestel Elektronik','Sanayi'),
    'VESBE':('Vestel Beyaz Eşya','Sanayi'),
    'ULKER':('Ülker Bisküvi','Sanayi'),
    'BRSAN':('Borusan Mannesmann','Sanayi'),
    'TRKCM':('Trakya Cam','Sanayi'),
    'SISE':('Şişe Cam','Sanayi'),
    'PRKAB':('Türk Prysmian Kablo','Sanayi'),
    'GUBRF':('Gübre Fabrikaları','Sanayi'),
    'ISDMR':('İskenderun Demir Çelik','Sanayi'),
    'ADEL':('Adel Kalemcilik','Sanayi'),
    'ERBOS':('Erbosan','Sanayi'),
    'SARKY':('Sarkuysan','Sanayi'),
    'TUPRS':('Tüpraş','Enerji'),
    'AYGAZ':('Aygaz','Enerji'),
    'AKSEN':('Aksa Enerji','Enerji'),
    'ZOREN':('Zorlu Enerji','Enerji'),
    'ENKAI':('Enka İnşaat','Enerji'),
    'ODAS':('Odaş Elektrik','Enerji'),
    'OSEN':('Osmanlı Elektrik','Enerji'),
    'THYAO':('Türk Hava Yolları','Havacılık'),
    'PGSUS':('Pegasus Havayolları','Havacılık'),
    'TAVHL':('TAV Havalimanları','Havacılık'),
    'CLEBI':('Çelebi Hava Servisi','Havacılık'),
    'BIMAS':('BİM Mağazalar','Perakende'),
    'SOKM':('ŞOK Marketler','Perakende'),
    'MAVI':('Mavi Giyim','Perakende'),
    'HEPSI':('Hepsiburada','Perakende'),
    'CRFSA':('CarrefourSA','Perakende'),
    'ASELS':('Aselsan','Savunma'),
    'LOGO':('Logo Yazılım','Teknoloji'),
    'INDES':('İndeks Bilgisayar','Teknoloji'),
    'KONTR':('Kontrolmatik','Teknoloji'),
    'KAREL':('Karel Elektronik','Teknoloji'),
    'NETAS':('Netaş Telekomünikasyon','Teknoloji'),
    'TCELL':('Turkcell','Telekomünikasyon'),
    'TTKOM':('Türk Telekom','Telekomünikasyon'),
    'TOASO':('Tofaş Oto Fabrikası','Otomotiv'),
    'FROTO':('Ford Otosan','Otomotiv'),
    'DOAS':('Doğuş Otomotiv','Otomotiv'),
    'ASUZU':('Anadolu Isuzu','Otomotiv'),
    'KARSN':('Karsan Otomotiv','Otomotiv'),
    'ANSGR':('Anadolu Sigorta','Sigorta'),
    'AKGRT':('Aksigorta','Sigorta'),
    'RAYSG':('Ray Sigorta','Sigorta'),
    'YKSGR':('Yapı Kredi Sigorta','Sigorta'),
    'ISGYO':('İş GYO','GYO'),
    'EKGYO':('Emlak Konut GYO','GYO'),
    'TOILE':('Torunlar GYO','GYO'),
    'ALGYO':('Alarko GYO','GYO'),
    'OZRGY':('Özderici GYO','GYO'),
    'ECILC':('Eczacıbaşı İlaç','Kimya'),
    'PETKM':('Petkim Petrokimya','Kimya'),
    'SODA':('Soda Sanayii','Kimya'),
    'ALKIM':('Alkim Kimya','Kimya'),
    'DEVA':('Deva Holding','Kimya'),
    'SELEC':('Selçuk Ecza','Kimya'),
    'AEFES':('Anadolu Efes','Gıda'),
    'CCOLA':('Coca-Cola İçecek','Gıda'),
    'TATGD':('Tat Gıda','Gıda'),
    'BANVT':('Banvit','Gıda'),
    'KERVT':('Kerevitaş Gıda','Gıda'),
    'AKCNS':('Akçansa Çimento','İnşaat'),
    'BOLUC':('Bolu Çimento','İnşaat'),
    'CIMSA':('Çimsa Çimento','İnşaat'),
    'TKFEN':('Tekfen Holding','İnşaat'),
    'OYAKC':('Oyak Çimento','İnşaat'),
    'GOLTS':('Göltaş Çimento','İnşaat'),
    'ADANA':('Adana Çimento (A)','İnşaat'),
    'BUCIM':('Bursa Çimento','İnşaat'),
    'MRDIN':('Mardin Çimento','İnşaat'),
    'PRKME':('Park Elektrik','Madencilik'),
    'KRDMA':('Kardemir (A)','Madencilik'),
}

FALLBACK = {
    'AKBNK':{'mu':0.00052,'sigma':0.0215,'beta':1.02,'price':67.85,'change':-0.32},
    'GARAN':{'mu':0.00054,'sigma':0.0225,'beta':1.08,'price':89.40,'change':1.23},
    'HALKB':{'mu':0.00031,'sigma':0.0268,'beta':1.15,'price':28.90,'change':-1.24},
    'ISCTR':{'mu':0.00044,'sigma':0.0198,'beta':0.98,'price':56.20,'change':0.54},
    'VAKBN':{'mu':0.00041,'sigma':0.0245,'beta':1.09,'price':33.25,'change':0.90},
    'YKBNK':{'mu':0.00043,'sigma':0.0232,'beta':1.05,'price':43.10,'change':1.78},
    'TSKB':{'mu':0.00038,'sigma':0.0210,'beta':0.92,'price':12.85,'change':0.47},
    'ALBRK':{'mu':0.00028,'sigma':0.0255,'beta':1.10,'price':8.42,'change':-0.60},
    'QNBFB':{'mu':0.00035,'sigma':0.0222,'beta':1.01,'price':14.18,'change':0.21},
    'KLNMA':{'mu':0.00035,'sigma':0.0245,'beta':1.05,'price':18.42,'change':-0.11},
    'KCHOL':{'mu':0.00052,'sigma':0.0178,'beta':0.85,'price':198.40,'change':0.82},
    'SAHOL':{'mu':0.00050,'sigma':0.0185,'beta':0.88,'price':67.20,'change':0.36},
    'DOHOL':{'mu':0.00042,'sigma':0.0205,'beta':0.95,'price':24.60,'change':1.10},
    'MGROS':{'mu':0.00048,'sigma':0.0205,'beta':0.89,'price':234.60,'change':1.45},
    'MPARK':{'mu':0.00055,'sigma':0.0225,'beta':0.98,'price':102.40,'change':2.15},
    'GLYHO':{'mu':0.00038,'sigma':0.0248,'beta':1.08,'price':34.20,'change':-0.85},
    'ISMEN':{'mu':0.00040,'sigma':0.0230,'beta':1.00,'price':28.60,'change':0.42},
    'EREGL':{'mu':0.00050,'sigma':0.0210,'beta':0.92,'price':46.54,'change':-0.45},
    'KRDMD':{'mu':0.00038,'sigma':0.0248,'beta':1.08,'price':18.72,'change':0.64},
    'TTRAK':{'mu':0.00060,'sigma':0.0195,'beta':0.87,'price':428.60,'change':1.20},
    'ARCLK':{'mu':0.00048,'sigma':0.0215,'beta':0.96,'price':148.40,'change':-0.31},
    'VESTL':{'mu':0.00042,'sigma':0.0252,'beta':1.10,'price':38.90,'change':0.77},
    'VESBE':{'mu':0.00045,'sigma':0.0238,'beta':1.05,'price':74.30,'change':-1.02},
    'ULKER':{'mu':0.00044,'sigma':0.0198,'beta':0.85,'price':92.30,'change':0.22},
    'BRSAN':{'mu':0.00042,'sigma':0.0225,'beta':0.95,'price':88.50,'change':-0.44},
    'TRKCM':{'mu':0.00046,'sigma':0.0208,'beta':0.92,'price':58.70,'change':0.65},
    'SISE':{'mu':0.00049,'sigma':0.0200,'beta':0.90,'price':62.40,'change':-0.18},
    'PRKAB':{'mu':0.00055,'sigma':0.0228,'beta':1.00,'price':124.20,'change':0.65},
    'GUBRF':{'mu':0.00040,'sigma':0.0230,'beta':0.98,'price':54.20,'change':1.45},
    'ISDMR':{'mu':0.00040,'sigma':0.0248,'beta':1.08,'price':24.80,'change':-0.44},
    'ADEL':{'mu':0.00048,'sigma':0.0228,'beta':0.98,'price':84.60,'change':0.46},
    'ERBOS':{'mu':0.00044,'sigma':0.0238,'beta':1.02,'price':48.20,'change':0.42},
    'SARKY':{'mu':0.00046,'sigma':0.0232,'beta':1.00,'price':74.40,'change':-0.27},
    'TUPRS':{'mu':0.00062,'sigma':0.0195,'beta':0.87,'price':180.20,'change':0.67},
    'AYGAZ':{'mu':0.00040,'sigma':0.0180,'beta':0.78,'price':92.60,'change':0.33},
    'AKSEN':{'mu':0.00048,'sigma':0.0228,'beta':1.02,'price':43.80,'change':1.54},
    'ZOREN':{'mu':0.00040,'sigma':0.0252,'beta':1.12,'price':18.94,'change':-0.73},
    'ENKAI':{'mu':0.00044,'sigma':0.0188,'beta':0.82,'price':32.76,'change':0.12},
    'ODAS':{'mu':0.00038,'sigma':0.0265,'beta':1.18,'price':31.25,'change':2.38},
    'OSEN':{'mu':0.00040,'sigma':0.0260,'beta':1.15,'price':21.80,'change':2.14},
    'THYAO':{'mu':0.00082,'sigma':0.0285,'beta':1.32,'price':298.60,'change':2.14},
    'PGSUS':{'mu':0.00065,'sigma':0.0312,'beta':1.18,'price':521.40,'change':-0.85},
    'TAVHL':{'mu':0.00058,'sigma':0.0258,'beta':1.14,'price':248.60,'change':1.64},
    'CLEBI':{'mu':0.00055,'sigma':0.0268,'beta':1.18,'price':183.20,'change':1.22},
    'BIMAS':{'mu':0.00042,'sigma':0.0168,'beta':0.72,'price':389.40,'change':0.22},
    'SOKM':{'mu':0.00040,'sigma':0.0222,'beta':0.94,'price':78.30,'change':-0.68},
    'MAVI':{'mu':0.00058,'sigma':0.0245,'beta':1.06,'price':184.60,'change':1.35},
    'HEPSI':{'mu':0.00070,'sigma':0.0348,'beta':1.48,'price':62.40,'change':3.87},
    'CRFSA':{'mu':0.00035,'sigma':0.0235,'beta':0.98,'price':34.82,'change':-0.45},
    'ASELS':{'mu':0.00065,'sigma':0.0248,'beta':1.12,'price':83.45,'change':3.21},
    'LOGO':{'mu':0.00072,'sigma':0.0295,'beta':1.22,'price':142.80,'change':-1.87},
    'INDES':{'mu':0.00060,'sigma':0.0278,'beta':1.20,'price':286.40,'change':1.14},
    'KONTR':{'mu':0.00065,'sigma':0.0288,'beta':1.24,'price':74.60,'change':1.87},
    'KAREL':{'mu':0.00055,'sigma':0.0305,'beta':1.30,'price':98.60,'change':-1.44},
    'NETAS':{'mu':0.00052,'sigma':0.0315,'beta':1.28,'price':94.60,'change':0.95},
    'TCELL':{'mu':0.00048,'sigma':0.0185,'beta':0.80,'price':82.40,'change':0.58},
    'TTKOM':{'mu':0.00040,'sigma':0.0175,'beta':0.75,'price':42.20,'change':0.14},
    'TOASO':{'mu':0.00044,'sigma':0.0215,'beta':0.96,'price':156.20,'change':-0.54},
    'FROTO':{'mu':0.00062,'sigma':0.0225,'beta':1.01,'price':942.50,'change':1.12},
    'DOAS':{'mu':0.00050,'sigma':0.0220,'beta':0.97,'price':148.80,'change':0.83},
    'ASUZU':{'mu':0.00052,'sigma':0.0235,'beta':1.03,'price':164.80,'change':0.92},
    'KARSN':{'mu':0.00042,'sigma':0.0248,'beta':1.08,'price':44.20,'change':1.42},
    'ANSGR':{'mu':0.00042,'sigma':0.0215,'beta':0.90,'price':52.40,'change':0.48},
    'AKGRT':{'mu':0.00040,'sigma':0.0205,'beta':0.88,'price':38.60,'change':-0.26},
    'RAYSG':{'mu':0.00038,'sigma':0.0228,'beta':0.98,'price':14.52,'change':0.69},
    'YKSGR':{'mu':0.00041,'sigma':0.0210,'beta':0.91,'price':22.40,'change':0.14},
    'ISGYO':{'mu':0.00035,'sigma':0.0232,'beta':0.98,'price':14.38,'change':0.84},
    'EKGYO':{'mu':0.00038,'sigma':0.0242,'beta':1.04,'price':12.64,'change':1.52},
    'TOILE':{'mu':0.00040,'sigma':0.0250,'beta':1.08,'price':28.90,'change':-0.69},
    'ALGYO':{'mu':0.00032,'sigma':0.0255,'beta':1.12,'price':16.72,'change':0.31},
    'OZRGY':{'mu':0.00028,'sigma':0.0272,'beta':1.18,'price':8.94,'change':-0.45},
    'ECILC':{'mu':0.00048,'sigma':0.0218,'beta':0.94,'price':64.80,'change':0.72},
    'PETKM':{'mu':0.00048,'sigma':0.0222,'beta':0.98,'price':34.60,'change':0.74},
    'SODA':{'mu':0.00052,'sigma':0.0210,'beta':0.92,'price':62.80,'change':0.58},
    'ALKIM':{'mu':0.00050,'sigma':0.0220,'beta':0.90,'price':78.60,'change':0.87},
    'DEVA':{'mu':0.00044,'sigma':0.0235,'beta':1.02,'price':38.20,'change':-0.52},
    'SELEC':{'mu':0.00040,'sigma':0.0225,'beta':0.96,'price':52.60,'change':0.34},
    'AEFES':{'mu':0.00050,'sigma':0.0192,'beta':0.85,'price':128.60,'change':0.44},
    'CCOLA':{'mu':0.00055,'sigma':0.0185,'beta':0.80,'price':194.80,'change':0.92},
    'TATGD':{'mu':0.00042,'sigma':0.0215,'beta':0.93,'price':42.80,'change':-0.37},
    'BANVT':{'mu':0.00038,'sigma':0.0235,'beta':1.02,'price':62.40,'change':1.18},
    'KERVT':{'mu':0.00042,'sigma':0.0238,'beta':1.04,'price':38.40,'change':0.63},
    'AKCNS':{'mu':0.00048,'sigma':0.0205,'beta':0.90,'price':128.40,'change':0.54},
    'BOLUC':{'mu':0.00040,'sigma':0.0218,'beta':0.95,'price':62.80,'change':-0.23},
    'CIMSA':{'mu':0.00045,'sigma':0.0210,'beta':0.92,'price':94.60,'change':0.88},
    'TKFEN':{'mu':0.00050,'sigma':0.0215,'beta':0.95,'price':98.40,'change':-0.28},
    'OYAKC':{'mu':0.00045,'sigma':0.0205,'beta':0.90,'price':54.80,'change':0.28},
    'GOLTS':{'mu':0.00038,'sigma':0.0230,'beta':1.00,'price':32.60,'change':0.56},
    'ADANA':{'mu':0.00038,'sigma':0.0225,'beta':0.98,'price':52.40,'change':0.35},
    'BUCIM':{'mu':0.00038,'sigma':0.0220,'beta':0.96,'price':58.20,'change':0.18},
    'MRDIN':{'mu':0.00036,'sigma':0.0228,'beta':0.98,'price':44.80,'change':0.34},
    'PRKME':{'mu':0.00045,'sigma':0.0248,'beta':1.08,'price':34.60,'change':0.73},
    'KRDMA':{'mu':0.00036,'sigma':0.0252,'beta':1.10,'price':16.80,'change':0.60},
}

# ════════════════════════════════════════════════════════
# VERİ ÇEKME
# ════════════════════════════════════════════════════════
_cache = {}

def hisse_verisi_cek(kod, donem='1y'):
    try:
        t  = yf.Ticker(f'{kod}.IS')
        df = t.history(period=donem, auto_adjust=True)
        if df.empty or len(df) < 20: raise ValueError()
        df['ret'] = np.log(df['Close']/df['Close'].shift(1))
        df = df.dropna()
        mu    = float(df['ret'].mean())
        sigma = float(df['ret'].std())
        price = float(df['Close'].iloc[-1])
        chg   = float((df['Close'].iloc[-1]/df['Close'].iloc[-2]-1)*100)
        try:
            bist = yf.Ticker('XU100.IS').history(period=donem, auto_adjust=True)
            bist['ret'] = np.log(bist['Close']/bist['Close'].shift(1))
            bist = bist.dropna()
            common = df['ret'].index.intersection(bist['ret'].index)
            if len(common) > 30:
                cv   = np.cov(df['ret'].loc[common], bist['ret'].loc[common])
                beta = float(cv[0,1]/cv[1,1])
            else:
                beta = FALLBACK.get(kod,{}).get('beta',1.0)
        except:
            beta = FALLBACK.get(kod,{}).get('beta',1.0)
        result = {'mu':mu,'sigma':sigma,'beta':round(beta,2),
                  'price':round(price,2),'change':round(chg,2),
                  'closes':df['Close'].round(2).tolist(),
                  'returns':df['ret'].tolist(),'kaynak':'Yahoo Finance ✓'}
        return result
    except:
        fb = FALLBACK.get(kod,{'mu':0.0005,'sigma':0.022,'beta':1.0,'price':50.0,'change':0.0})
        return {**fb,'closes':[],'returns':[],'kaynak':'Fallback ⚠'}

# ════════════════════════════════════════════════════════
# RİSK HESAPLAMA
# ════════════════════════════════════════════════════════
def hesapla(hisseler, w_raw, pv, guven, hz, mc_n, donem, opt_metot):
    w = np.array(w_raw)/np.array(w_raw).sum()

    # Veri
    meta = {}
    for k in hisseler:
        v = hisse_verisi_cek(k, donem)
        v['sirket'] = BIST100.get(k,(k,))[0]
        v['sektor']  = BIST100.get(k,('','-'))[1]
        meta[k] = v

    mu_v   = np.array([meta[k]['mu']    for k in hisseler])
    sig_v  = np.array([meta[k]['sigma'] for k in hisseler])
    beta_v = np.array([meta[k].get('beta',1.0) for k in hisseler])

    # Korelasyon
    rets = [meta[k].get('returns',[]) for k in hisseler]
    if all(len(r)>30 for r in rets):
        ml  = min(len(r) for r in rets)
        dff = pd.DataFrame({k: meta[k]['returns'][-ml:] for k in hisseler})
        corr= dff.corr()
    else:
        n = len(hisseler)
        c = np.full((n,n),0.35); np.fill_diagonal(c,1.0)
        corr = pd.DataFrame(c,index=hisseler,columns=hisseler)

    cov = np.outer(sig_v,sig_v)*corr.values
    p_mu  = float(w@mu_v)
    p_sig = float(np.sqrt(w@cov@w))
    p_beta= float(w@beta_v)

    # Parametrik VaR
    z     = float(-norm.ppf(1-guven))
    sqrtT = float(np.sqrt(hz))
    d_var = pv*(p_mu*hz - z*p_sig*sqrtT)
    g_var = pv*(p_mu - z*p_sig)
    vp    = abs(d_var)/pv*100
    cvar  = pv*p_sig*float(norm.pdf(z))/(1-guven) - pv*p_mu
    sharpe= (p_mu*252-0.35)/(p_sig*np.sqrt(252))

    # Monte Carlo
    np.random.seed(42)
    pl   = pv*(p_mu*hz + p_sig*sqrtT*np.random.standard_normal(mc_n))
    pl_s = np.sort(pl)
    mc_v = abs(float(pl_s[int((1-guven)*mc_n)]))
    mc_k = abs(float(pl_s[0]))
    mc_m = float(np.median(pl_s))

    # Senaryo
    sen = {
        'boga': float(pv*(p_mu+2*p_sig)*sqrtT),
        'baz':  float(pv*p_mu*hz),
        'ayi':  float(pv*(p_mu-2*p_sig)*sqrtT),
        'kriz': float(pv*(p_mu-3.5*p_sig)*sqrtT),
    }

    # GARCH
    garch = {}
    ret0 = meta[hisseler[0]].get('returns', [])
    if not ret0 or len(ret0) < 60:
        np.random.seed(42)
        sig0 = meta[hisseler[0]].get('sigma', 0.02)
        mu0  = meta[hisseler[0]].get('mu', 0.0005)
        ret0 = (np.random.normal(mu0, sig0, 500)).tolist()
    if ret0 and len(ret0) > 60:
        try:
            from arch import arch_model
            r   = pd.Series(ret0) * 100
            r   = r.dropna()
            res = arch_model(r, vol='Garch', p=1, q=1, dist='normal', rescale=False).fit(disp='off', show_warning=False)
            fc  = res.forecast(horizon=hz, reindex=False)
            garch = {
                'dates':    [str(d.date()) for d in res.conditional_volatility.index],
                'cond_vol': (res.conditional_volatility / 100).tolist(),
                'omega':    float(res.params.get('omega', 0)),
                'alpha':    float(res.params.get('alpha[1]', 0)),
                'beta_g':   float(res.params.get('beta[1]', 0)),
                'tahmin':   float(np.sqrt(fc.variance.values[-1, :].mean())) / 100,
            }
        except Exception as eg:
            garch = {'hata': str(eg)}

    # Optimizasyon
    opt = {}
    if all(len(r)>30 for r in rets):
        try:
            from pypfopt import EfficientFrontier, risk_models, expected_returns
            mu_y  = expected_returns.mean_historical_return(dff,returns_data=True,frequency=252)
            cov_y = risk_models.sample_cov(dff,returns_data=True,frequency=252)
            ef = EfficientFrontier(mu_y,cov_y,weight_bounds=(0,0.4))
            if opt_metot=='sharpe': ef.max_sharpe(risk_free_rate=0.35)
            else: ef.min_volatility()
            wc  = ef.clean_weights()
            prf = ef.portfolio_performance(verbose=False,risk_free_rate=0.35)
            opt = {'w':wc,'ret':round(prf[0]*100,2),'vol':round(prf[1]*100,2),'sharpe':round(prf[2],3)}
        except: pass

    # Katkı tablosu
    rows = []
    for i,k in enumerate(hisseler):
        hvc = pv*w[i]*(mu_v[i]-z*sig_v[i])
        rows.append({
            'Kod':k,'Şirket':meta[k]['sirket'],'Sektör':meta[k]['sektor'],
            'Fiyat ₺':meta[k]['price'],'Değişim %':round(meta[k]['change'],2),
            'Ağırlık %':round(w[i]*100,1),
            'μ/gün %':round(mu_v[i]*100,4),'σ/gün %':round(sig_v[i]*100,4),
            'Beta':round(float(beta_v[i]),2),'VaR Katkı ₺':round(abs(hvc),0),
            'Yıllık Vol %':round(sig_v[i]*np.sqrt(252)*100,1),
            'Kaynak':meta[k]['kaynak']
        })

    return {
        'p_mu':p_mu,'p_sig':p_sig,'p_beta':p_beta,'z':z,
        'd_var':abs(d_var),'g_var':abs(g_var),'vp':vp,
        'cvar':abs(cvar),'sharpe':sharpe,
        'yillik_mu':p_mu*252*100,'yillik_vol':p_sig*np.sqrt(252)*100,
        'mc_var':mc_v,'mc_kotu':mc_k,'mc_medyan':mc_m,
        'pl_s':pl_s.tolist(),'sen':sen,'katki':rows,
        'corr':corr,'meta':meta,'w':w.tolist(),
        'garch':garch,'opt':opt,
    }

# ════════════════════════════════════════════════════════
# RENKLER VE CSS
# ════════════════════════════════════════════════════════
BG=     '#0a0e1a'; CARD='#131c2e'; PANEL='#111827'
ACCENT= '#63b3ed'; DANGER='#fc8181'; SUCCESS='#68d391'
WARNING='#f6e05e'; PURPLE='#b794f4'; TEXT='#e2e8f0'
MUTED=  '#718096'; GRID='#1a2235'
COLORS= ['#63b3ed','#68d391','#f6e05e','#fc8181','#b794f4',
         '#76e4f7','#fbb6ce','#9ae6b4','#fed7aa','#c3dafe',
         '#fefcbf','#bee3f8','#c6f6d5','#fed7d7','#e9d8fd']

PL = dict(paper_bgcolor=CARD,plot_bgcolor=CARD,
          font=dict(family='IBM Plex Mono',color=TEXT,size=11),
          margin=dict(l=50,r=20,t=40,b=40),
          xaxis=dict(gridcolor=GRID,zerolinecolor=GRID),
          yaxis=dict(gridcolor=GRID,zerolinecolor=GRID),
          legend=dict(bgcolor=CARD,bordercolor=GRID,borderwidth=1,font=dict(size=10)))

SEKTORLER = sorted(set(v[1] for v in BIST100.values()))

def ftl(v): return f'₺{abs(v):,.0f}'
def gc(fig,h=320):
    return dcc.Graph(figure=fig,config={'displayModeBar':True,'displaylogo':False},
                     style={'height':f'{h}px'})
def krt(lbl,val,sub,color,cls):
    return html.Div([html.Div(lbl,className='metric-label'),
                     html.Div(val,className='metric-value',style={'color':color}),
                     html.Div(sub,className='metric-sub')],className=f'metric-card {cls}')

CSS = '''
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root{--bg:#0a0e1a;--bg2:#0f1724;--bg3:#1a2235;--card:#131c2e;--panel:#111827;
  --border:rgba(99,179,237,.14);--border2:rgba(99,179,237,.28);
  --accent:#63b3ed;--danger:#fc8181;--success:#68d391;--warning:#f6e05e;--purple:#b794f4;
  --text:#e2e8f0;--text2:#a0aec0;--text3:#718096;
  --mono:"IBM Plex Mono",monospace;--sans:"IBM Plex Sans",sans-serif;}
*{margin:0;padding:0;box-sizing:border-box;}
body,.app-wrap{background:var(--bg)!important;color:var(--text);font-family:var(--sans);min-height:100vh;}
.header{background:var(--bg2);border-bottom:1px solid var(--border2);padding:0 2rem;position:sticky;top:0;z-index:200;box-shadow:0 2px 20px rgba(0,0,0,.4);}
.header-inner{max-width:1500px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;height:60px;}
.logo{display:flex;align-items:center;gap:12px;}
.logo-icon{width:38px;height:38px;border-radius:9px;background:linear-gradient(135deg,#63b3ed,#2b6cb0);display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:13px;font-weight:700;color:#fff;}
.logo-text{font-family:var(--mono);font-size:14px;font-weight:600;letter-spacing:.04em;}
.logo-sub{font-size:10px;color:var(--text3);font-family:var(--mono);}
.header-right{display:flex;align-items:center;gap:10px;}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--success);box-shadow:0 0 7px var(--success);animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.live-txt{font-family:var(--mono);font-size:11px;color:var(--success);}
.badge{font-family:var(--mono);font-size:11px;padding:4px 12px;border:1px solid var(--border2);border-radius:20px;color:var(--accent);}
.badge-sm{font-family:var(--mono);font-size:10px;color:var(--text3);}
.main-grid{max-width:1500px;margin:0 auto;padding:1.5rem 2rem;display:grid;grid-template-columns:310px 1fr;gap:1.5rem;align-items:start;}
.panel{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:1.2rem;}
.panel-header{padding:.9rem 1.2rem;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}
.panel-title{font-size:11px;font-weight:600;font-family:var(--mono);color:var(--accent);letter-spacing:.08em;text-transform:uppercase;}
.panel-body{padding:1.1rem;}
.search-input{width:100%;background:var(--bg3)!important;border:1px solid var(--border)!important;border-radius:8px;padding:9px 12px;color:var(--text)!important;font-family:var(--mono);font-size:12px;outline:none;margin-bottom:.7rem;transition:border-color .2s;}
.search-input:focus{border-color:var(--accent)!important;}
.search-input::placeholder{color:var(--text3);}
.pills{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:.7rem;}
.pill{font-size:10px;font-family:var(--mono);padding:3px 8px;border:1px solid var(--border);border-radius:20px;background:transparent;color:var(--text3);cursor:pointer;transition:all .18s;white-space:nowrap;}
.pill:hover,.pill.active{border-color:var(--accent);color:var(--accent);background:rgba(99,179,237,.08);}
.stock-badge{font-size:10px;color:var(--text3);font-family:var(--mono);margin-bottom:5px;}
.stock-list{max-height:280px;overflow-y:auto;display:flex;flex-direction:column;gap:3px;}
.stock-list::-webkit-scrollbar{width:3px;}
.stock-list::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
.stock-item{display:flex;align-items:center;justify-content:space-between;padding:7px 10px;border-radius:7px;cursor:pointer;border:1px solid transparent;background:var(--bg3);transition:all .13s;}
.stock-item:hover{border-color:var(--border2);}
.stock-item.selected{border-color:var(--accent);background:rgba(99,179,237,.07);}
.stock-code{font-family:var(--mono);font-size:12px;font-weight:600;}
.stock-name{font-size:10px;color:var(--text3);margin-top:1px;}
.stock-check{width:16px;height:16px;border-radius:3px;border:1.5px solid var(--border2);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:10px;transition:all .13s;}
.stock-check.checked{background:var(--accent);border-color:var(--accent);color:white;}
.chips{display:flex;flex-wrap:wrap;gap:5px;min-height:24px;margin-bottom:.8rem;}
.chip{display:flex;align-items:center;gap:4px;padding:3px 8px;border-radius:5px;background:rgba(99,179,237,.1);border:1px solid rgba(99,179,237,.25);font-family:var(--mono);font-size:11px;color:var(--accent);}
.chip-x{cursor:pointer;color:var(--text3);font-size:11px;}
.chip-x:hover{color:var(--danger);}
.empty{color:var(--text3);font-size:11px;font-style:italic;}
.weight-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:.6rem;}
.weight-label{font-size:10px;color:var(--text3);font-family:var(--mono);}
.eq-btn{padding:5px 12px;background:var(--bg3);border:1px solid var(--border2);border-radius:6px;cursor:pointer;font-family:var(--mono);font-size:10px;color:var(--accent);}
.eq-btn:hover{background:rgba(99,179,237,.1);}
.weight-row{display:flex;align-items:center;gap:7px;margin-bottom:7px;}
.weight-code{font-family:var(--mono);font-size:11px;color:var(--accent);min-width:52px;}
.weight-val{font-family:var(--mono);font-size:11px;color:var(--text2);min-width:36px;text-align:right;}
.weight-slider{flex:1;}
.weight-total{font-family:var(--mono);font-size:12px;margin-top:6px;text-align:right;}
.weight-total.ok{color:var(--success);}.weight-total.err{color:var(--danger);}
.param-row{margin-bottom:1rem;}
.param-label{font-size:11px;color:var(--text2);font-family:var(--mono);margin-bottom:6px;display:flex;justify-content:space-between;}
.param-val{color:var(--accent);font-weight:500;}
.rc-slider-track{background-color:var(--accent)!important;}
.rc-slider-handle{border-color:var(--accent)!important;background:var(--accent)!important;box-shadow:none!important;width:14px!important;height:14px!important;margin-top:-5px!important;}
.rc-slider-rail{background-color:var(--bg3)!important;}
.Select-control,.Select-menu-outer{background:var(--bg3)!important;border:1px solid var(--border)!important;color:var(--text)!important;font-family:var(--mono)!important;font-size:12px!important;border-radius:7px!important;}
.Select-value-label,.Select-placeholder{color:var(--text)!important;}
.Select-option{background:var(--bg3)!important;color:var(--text)!important;font-size:12px!important;}
.Select-option:hover{background:var(--bg2)!important;}
.run-btn{width:100%;padding:13px;background:linear-gradient(135deg,#2b6cb0,#2c5282);border:none;border-radius:9px;cursor:pointer;font-family:var(--mono);font-size:13px;font-weight:700;color:#fff;letter-spacing:.06em;transition:all .2s;margin-top:.4rem;}
.run-btn:hover:not(:disabled){background:linear-gradient(135deg,#3182ce,#2b6cb0);transform:translateY(-1px);}
.run-btn:disabled{opacity:.4;cursor:not-allowed;}
.metrics-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:.9rem;margin-bottom:1.2rem;}
.metric-card{background:var(--card);border:1px solid var(--border);border-radius:11px;padding:1rem;position:relative;overflow:hidden;}
.metric-card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;}
.metric-card.red::before{background:var(--danger);}.metric-card.yellow::before{background:var(--warning);}
.metric-card.blue::before{background:var(--accent);}.metric-card.green::before{background:var(--success);}
.metric-card.purple::before{background:var(--purple);}
.metric-label{font-size:10px;color:var(--text3);font-family:var(--mono);text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px;}
.metric-value{font-size:19px;font-weight:700;font-family:var(--mono);line-height:1;}
.metric-sub{font-size:10px;color:var(--text3);margin-top:4px;font-family:var(--mono);}
.custom-tabs{border-bottom:1px solid var(--border)!important;margin-bottom:1.2rem;}
.tab{background:transparent!important;border:none!important;border-bottom:2px solid transparent!important;color:var(--text3)!important;font-family:var(--mono)!important;font-size:11px!important;text-transform:uppercase!important;letter-spacing:.05em!important;padding:8px 14px!important;}
.tab-active{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;background:transparent!important;}
.alert{padding:10px 14px;border-radius:8px;font-size:11px;margin-bottom:1rem;font-family:var(--mono);}
.alert-info{background:rgba(99,179,237,.08);border:1px solid rgba(99,179,237,.2);color:var(--accent);}
.alert-danger{background:rgba(252,129,129,.07);border:1px solid rgba(252,129,129,.28);color:var(--danger);}
.alert-warning{background:rgba(246,224,94,.07);border:1px solid rgba(246,224,94,.25);color:var(--warning);}
.scenario-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;}
.scenario-card{background:var(--card);border:1px solid var(--border);border-radius:9px;padding:.9rem;}
.scenario-title{font-size:10px;font-family:var(--mono);color:var(--text2);margin-bottom:5px;text-transform:uppercase;}
.scenario-val{font-size:18px;font-weight:700;font-family:var(--mono);}
.formula-box{margin-top:.8rem;padding:.8rem;background:var(--bg3);border-radius:8px;font-family:var(--mono);font-size:11px;color:var(--text3);line-height:1.9;}
.placeholder{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4rem;gap:14px;text-align:center;background:var(--card);border:1px solid var(--border);border-radius:12px;}
.place-icon{font-size:52px;opacity:.12;}.place-title{font-size:14px;color:var(--text2);font-family:var(--mono);}
.place-sub{font-size:12px;color:var(--text3);max-width:380px;line-height:1.6;}
@media(max-width:1100px){.main-grid{grid-template-columns:1fr;}.metrics-grid{grid-template-columns:repeat(2,1fr);}.scenario-grid{grid-template-columns:repeat(2,1fr);}}
'''

# ════════════════════════════════════════════════════════
# DASH UYGULAMASI BAŞLATMA
# ════════════════════════════════════════════════════════
app = dash.Dash(__name__, title='BIST 100 VaR Platform', suppress_callback_exceptions=True)
server = app.server

app.index_string = '''
<!DOCTYPE html><html><head>
{%metas%}<title>{%title%}</title>{%favicon%}
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>''' + CSS + '''</style>
{%css%}</head><body>{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>
'''

app.layout = html.Div([
    # HEADER
    html.Div([html.Div([
        html.Div([html.Div('VaR',className='logo-icon'),
                  html.Div([html.Div('BIST 100 · Portfolio Risk Analytics',className='logo-text'),
                            html.Div('Bloomberg PORT · yfinance · GARCH · PyPortfolioOpt',className='logo-sub')])],className='logo'),
        html.Div([html.Div(className='live-dot'),html.Div('15dk Gecikmeli',className='live-txt'),
                  html.Div('Akademik Proje',className='badge')],className='header-right'),
    ],className='header-inner')],className='header'),

    html.Div([
        # SOL PANEL
        html.Div([
            # Hisse Seçimi
            html.Div([html.Div([html.Span('📊 Hisse Seçimi',className='panel-title'),
                                html.Span('0/15',id='sel-cnt',className='badge-sm')],className='panel-header'),
                      html.Div([dcc.Input(id='srch',type='text',debounce=True,
                                          placeholder='Hisse ara... THYAO, GARAN...',className='search-input'),
                                html.Div([html.Button('Tümü',id='p-tumu',className='pill active',n_clicks=0,**{'data-sector':'Tümü'})]+
                                         [html.Button(s,className='pill',n_clicks=0,
                                                      id=f'p-{i}',**{'data-sector':s}) for i,s in enumerate(SEKTORLER)],
                                         className='pills',id='pill-wrap'),
                                html.Div('',id='h-cnt',className='stock-badge'),
                                html.Div(id='h-list',className='stock-list'),
                               ],className='panel-body')],className='panel'),
            # Portföy
            html.Div([html.Div([html.Span('💼 Portföy (maks 15)',className='panel-title')],className='panel-header'),
                      html.Div([html.Div([html.Span('Hisse seçilmedi',className='empty')],id='chips',className='chips'),
                                html.Div(id='w-bolum',style={'display':'none'},
                                         children=[html.Div([html.Span('Ağırlıklar (%)',className='weight-label'),
                                                             html.Button('⚖ Eşit',id='esit-btn',n_clicks=0,className='eq-btn')],
                                                            className='weight-header'),
                                                   html.Div(id='w-sliders'),
                                                   html.Div('Toplam: %100',id='w-toplam',className='weight-total ok')])],
                               className='panel-body')],className='panel'),
            # Parametreler
            html.Div([html.Div([html.Span('⚙️ Parametreler',className='panel-title')],className='panel-header'),
                      html.Div([
                          html.Div([html.Div([html.Span('Portföy Değeri (₺)'),html.Span('1.000.000 ₺',id='pv-l',className='param-val')],className='param-label'),
                                    dcc.Slider(id='pv',min=100_000,max=10_000_000,step=100_000,value=1_000_000,marks=None,tooltip={'always_visible':False})],className='param-row'),
                          html.Div([html.Div([html.Span('Güven Seviyesi'),html.Span('%99',id='ci-l',className='param-val')],className='param-label'),
                                    dcc.Slider(id='ci',min=90,max=99,step=1,value=99,marks=None,tooltip={'always_visible':False})],className='param-row'),
                          html.Div([html.Div([html.Span('Zaman Ufku'),html.Span('1 Gün',id='hz-l',className='param-val')],className='param-label'),
                                    dcc.Slider(id='hz',min=1,max=30,step=1,value=1,marks=None,tooltip={'always_visible':False})],className='param-row'),
                          html.Div([html.Div([html.Span('Monte Carlo'),html.Span('10.000',id='mc-l',className='param-val')],className='param-label'),
                                    dcc.Slider(id='mc',min=1000,max=50_000,step=1000,value=10_000,marks=None,tooltip={'always_visible':False})],className='param-row'),
                          html.Div([html.Div('Veri Dönemi',className='param-label'),
                                    dcc.Dropdown(id='donem',options=[{'label':'6 Ay','value':'6mo'},{'label':'1 Yıl','value':'1y'},{'label':'2 Yıl','value':'2y'}],
                                                 value='1y',clearable=False,className='dark-dd')],className='param-row'),
                          html.Div([html.Div('Optimizasyon',className='param-label'),
                                    dcc.Dropdown(id='opt',options=[{'label':'Maks Sharpe','value':'sharpe'},{'label':'Min Volatilite','value':'minvol'}],
                                                 value='sharpe',clearable=False,className='dark-dd')],className='param-row'),
                      ],className='panel-body')],className='panel'),
        ],className='sidebar'),

        # ANA İÇERİK
        html.Div([html.Div(id='sonuc'),
                  dcc.Store(id='s-sel',data=[]),
                  dcc.Store(id='s-w',  data={}),
                  dcc.Store(id='s-sek',data='Tümü'),
                  dcc.Store(id='s-res',data={}),
                 ],className='content'),
    ],className='main-grid'),
],className='app-wrap')

# ── Slider etiketleri ─────────────────────────────────────────
@app.callback(Output('pv-l','children'),Input('pv','value'))
def f1(v): return f'{v:,.0f} ₺'.replace(',','.')
@app.callback(Output('ci-l','children'),Input('ci','value'))
def f2(v): return f'%{v}'
@app.callback(Output('hz-l','children'),Input('hz','value'))
def f3(v): return f'{v} Gün'
@app.callback(Output('mc-l','children'),Input('mc','value'))
def f4(v): return f'{v:,.0f}'.replace(',','.')

# ── Aktif sektör ──────────────────────────────────────────────
@app.callback(Output('s-sek','data'),
              [Input('p-tumu','n_clicks')]+[Input(f'p-{i}','n_clicks') for i in range(len(SEKTORLER))],
              prevent_initial_call=True)
def sek(*args):
    ctx = dash.callback_context
    if not ctx.triggered: return 'Tümü'
    bid = ctx.triggered[0]['prop_id'].split('.')[0]
    if bid=='p-tumu': return 'Tümü'
    try:
        idx = int(bid.split('-')[1])
        return SEKTORLER[idx]
    except: return 'Tümü'

# ── Hisse listesi ─────────────────────────────────────────────
@app.callback(Output('h-list','children'),Output('h-cnt','children'),
              Input('srch','value'),Input('s-sel','data'),Input('s-sek','data'))
def hlist(ara,sel,sek):
    q=( ara or '').lower(); s=sek or 'Tümü'; sel=sel or []
    fil=[(k,v) for k,v in BIST100.items() if (s=='Tümü' or v[1]==s) and (q in k.lower() or q in v[0].lower())]
    items=[]
    for k,v in fil:
        on=k in sel; fb=FALLBACK.get(k,{}); chg=fb.get('change',0)
        cc=SUCCESS if chg>=0 else DANGER
        items.append(html.Div([
            html.Div([html.Div(k,className='stock-code'),html.Div(v[0],className='stock-name')]),
            html.Div([html.Div([html.Div(f"{'+' if chg>=0 else ''}{chg:.2f}%",style={'color':cc,'fontFamily':'IBM Plex Mono','fontSize':'11px'}),
                                html.Div(f"β{fb.get('beta',1.0):.2f}",style={'color':MUTED,'fontSize':'9px'})],style={'textAlign':'right'}),
                      html.Div('✓' if on else '',className=f"stock-check {'checked' if on else ''}")],
                     style={'display':'flex','alignItems':'center','gap':'6px'})],
            className=f"stock-item {'selected' if on else ''}",
            id={'type':'si','index':k},n_clicks=0,style={'cursor':'pointer'}))
    return items, f'{len(fil)} hisse'

# ── Hisse toggle ──────────────────────────────────────────────
@app.callback(Output('s-sel','data'),Output('s-w','data'),
              Input({'type':'si','index':ALL},'n_clicks'),
              State({'type':'si','index':ALL},'id'),
              State('s-sel','data'),State('s-w','data'),prevent_initial_call=True)
def tog(clicks,ids,sel,ws):
    ctx=dash.callback_context
    if not ctx.triggered or not any(c for c in (clicks or [])): return sel or [],ws or {}
    k=json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index']
    sel=list(sel or []); ws=dict(ws or {})
    if k in sel: sel.remove(k); ws.pop(k,None)
    else:
        if len(sel)>=15: return sel,ws
        sel.append(k)
    eq = round(100/len(sel), 1) if sel else 0
    for x in sel:
        if x not in ws:
            ws[x] = eq
    return sel, ws

# ── Chips + sliderlar ─────────────────────────────────────────
@app.callback(
    Output('chips','children'),Output('w-bolum','style'),Output('w-sliders','children'),
    Output('w-toplam','children'),Output('w-toplam','className'),
    Output('sel-cnt','children'),Output('run','disabled'),
    Input('s-sel','data'),Input('s-w','data'))
def port_ui(sel,ws):
    sel=sel or []; ws=ws or {}
    if not sel:
        return ([html.Span('Hisse seçilmedi',className='empty')],
                {'display':'none'},[],'',' weight-total','0/15',True)
    chips=[html.Div([k,html.Span('✕',className='chip-x',id={'type':'cx','index':k},n_clicks=0)],className='chip') for k in sel]
    sliders=[]
    for k in sel:
        val=ws.get(k,round(100/len(sel),1))
        sliders.append(html.Div([
            html.Span(k,className='weight-code'),
            dcc.Slider(id={'type':'wsl','index':k},min=0,max=100,step=0.5,value=val,marks=None,tooltip={'always_visible':False},className='weight-slider'),
            html.Span(f'{val:.1f}%',className='weight-val',id={'type':'wv','index':k}),
        ],className='weight-row'))
    t=sum(ws.get(k,0) for k in sel); ok=abs(t-100)<0.6
    return (chips,{'display':'block'},sliders,
            f"Toplam: %{t:.1f} {'✓' if ok else '→ 100 olmalı'}",
            'weight-total ok' if ok else 'weight-total err',
            f'{len(sel)}/15',False)

# ── Ağırlık slider → store ────────────────────────────────────
@app.callback(Output('s-w','data',allow_duplicate=True),
              Input({'type':'wsl','index':ALL},'value'),
              State({'type':'wsl','index':ALL},'id'),prevent_initial_call=True)
def w_up(vals,ids):
    if not ids: return {}
    return {ids[i]['index']:vals[i] for i in range(len(ids))}

# ── Eşit ──────────────────────────────────────────────────────
@app.callback(Output('s-w','data',allow_duplicate=True),
              Input('esit-btn','n_clicks'),State('s-sel','data'),prevent_initial_call=True)
def esit(n,sel):
    if not sel: return {}
    eq=round(100/len(sel),1)
    return {k:eq for k in sel}

# ── Chip sil ──────────────────────────────────────────────────
@app.callback(Output('s-sel','data',allow_duplicate=True),
              Input({'type':'cx','index':ALL},'n_clicks'),
              State({'type':'cx','index':ALL},'id'),State('s-sel','data'),prevent_initial_call=True)
def cx_sil(clicks,ids,sel):
    ctx=dash.callback_context
    if not ctx.triggered or not any(c for c in (clicks or [])): return sel or []
    k=json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index']
    return [x for x in (sel or []) if x!=k]

# ── ANALİZ ────────────────────────────────────────────────────
@app.callback(Output('sonuc','children'),
              Input('s-sel','data'),
              Input('s-w','data'),
              State('pv','value'),State('ci','value'),State('hz','value'),
              State('mc','value'),State('donem','value'),State('opt','value'),
              prevent_initial_call=True)
def analiz(sel,ws,pv,ci,hz,mc_n,donem,opt_metot):
    if not sel or len(sel)<2: return html.Div('En az 2 hisse seçin.',style={'color':'#718096','fontFamily':'IBM Plex Mono','padding':'2rem','textAlign':'center'})
    h=sel[:15]; guven=ci/100
    w_raw=[ws.get(k,100/len(h)) for k in h]
    R=hesapla(h,w_raw,pv,guven,hz,mc_n,donem,opt_metot)
    w=R['w']; meta=R['meta']; sen=R['sen']; garch=R['garch']; opt=R['opt']
    vp=R['vp']; vc=DANGER if vp>5 else WARNING if vp>2 else SUCCESS
    rl='🔴 YÜKSEK' if vp>5 else '🟡 ORTA' if vp>2 else '🟢 DÜŞÜK'

    # Grafikler
    mu,sig=R['p_mu'],R['p_sig']
    x=np.linspace(mu-4.2*sig,mu+4.2*sig,400)
    vx=mu+norm.ppf(1-guven)*sig
    f1=go.Figure()
    f1.add_trace(go.Scatter(x=x[x<=vx]*100,y=norm.pdf(x[x<=vx],mu,sig)/100,fill='tozeroy',fillcolor='rgba(252,129,129,0.4)',line=dict(color='rgba(0,0,0,0)'),name='Kayıp'))
    f1.add_trace(go.Scatter(x=x[x>vx]*100,y=norm.pdf(x[x>vx],mu,sig)/100,fill='tozeroy',fillcolor='rgba(99,179,237,0.2)',line=dict(color='rgba(0,0,0,0)'),name='Güvenli'))
    f1.add_trace(go.Scatter(x=x*100,y=norm.pdf(x,mu,sig)/100,line=dict(color=ACCENT,width=2.5),showlegend=False))
    f1.add_vline(x=vx*100,line=dict(color=DANGER,width=2,dash='dash'),annotation_text=f'VaR %{vp:.2f}',annotation_font_color=DANGER)
    f1.update_layout(**PL,title=f'Getiri Dağılımı  μ={mu*100:.4f}%  σ={sig*100:.4f}%',xaxis_title='Günlük Getiri (%)',yaxis_title='Olasılık Yoğunluğu')

    pl=np.array(R['pl_s']); ve=-R['mc_var']
    f2=go.Figure()
    f2.add_trace(go.Histogram(x=pl[pl<ve]/1e3,nbinsx=30,marker_color=DANGER,opacity=0.7,name='Kayıp'))
    f2.add_trace(go.Histogram(x=pl[pl>=ve]/1e3,nbinsx=30,marker_color=ACCENT,opacity=0.5,name='Kar'))
    f2.add_vline(x=ve/1e3,line=dict(color=DANGER,width=2,dash='dash'),annotation_text=f'MC VaR {ftl(R["mc_var"])}',annotation_font_color=DANGER)
    f2.update_layout(**PL,barmode='overlay',title=f'Monte Carlo P&L — {mc_n:,} Simülasyon',xaxis_title='P&L (Bin ₺)',yaxis_title='Simülasyon Sayısı')

    f3a=go.Figure(go.Pie(labels=h,values=[round(w[i]*100,1) for i in range(len(h))],hole=0.60,
        marker=dict(colors=COLORS[:len(h)],line=dict(color=BG,width=2))))
    f3a.update_layout(**PL,title='Portföy Ağırlıkları',annotations=[dict(text='Portföy',x=0.5,y=0.5,font_size=13,showarrow=False,font_color=TEXT)])

    mu_v=np.array([meta[k]['mu'] for k in h]); sig_v=np.array([meta[k]['sigma'] for k in h])
    vols=[sig_v[i]*np.sqrt(252)*100 for i in range(len(h))]
    f3b=go.Figure(go.Bar(x=h,y=vols,marker_color=[DANGER if v>40 else WARNING if v>25 else SUCCESS for v in vols],
        text=[f'%{v:.1f}' for v in vols],textposition='outside'))
    f3b.update_layout(**PL,showlegend=False,title='Yıllık Volatilite',yaxis_title='%')

    f4=go.Figure()
    for i,k in enumerate(h[:8]):
        kap=meta[k].get('closes',[])
        if kap:
            arr=np.array(kap)/kap[0]*100
            f4.add_trace(go.Scatter(y=arr.tolist(),name=k,line=dict(color=COLORS[i%len(COLORS)],width=1.8)))
    f4.add_hline(y=100,line=dict(color=MUTED,width=1,dash='dot'))
    f4.update_layout(**PL,title=f'Normalize Fiyat — {donem}',xaxis_title='Gün',yaxis_title='Endeks (Baz=100)')

    cv=R['corr'].values
    f5=go.Figure(go.Heatmap(z=cv,x=h,y=h,colorscale='RdYlGn',zmin=-1,zmax=1,
        text=[[f'{cv[i][j]:.2f}' for j in range(len(h))] for i in range(len(h))],
        texttemplate='%{text}',textfont={'size':9},colorbar=dict(title='ρ',tickfont=dict(color=TEXT))))
    f5.update_layout(**PL,title='Korelasyon Matrisi')

    f6=go.Figure()
    if garch.get('dates'):
        f6.add_trace(go.Scatter(x=garch['dates'],y=[v*100 for v in garch['cond_vol']],line=dict(color=PURPLE,width=1.5),name='Koşullu Volatilite'))
        f6.update_layout(**PL,title=f'GARCH(1,1) — {h[0]}',xaxis_title='Tarih',yaxis_title='Günlük Vol (%)')
    else:
        f6.add_annotation(text='GARCH için yfinance verisi gerekli',x=0.5,y=0.5,showarrow=False,font=dict(color=MUTED,size=13))
        f6.update_layout(**PL,title='GARCH(1,1)')

    f7=go.Figure()
    if opt.get('w'):
        lb=list(opt['w'].keys()); ov=[opt['w'][k]*100 for k in lb]
        cv2=[w[h.index(k)]*100 if k in h else 0 for k in lb]
        f7.add_trace(go.Bar(name='Mevcut',x=lb,y=cv2,marker_color=ACCENT))
        f7.add_trace(go.Bar(name='Optimal',x=lb,y=ov,marker_color=SUCCESS))
        f7.update_layout(**PL,barmode='group',title=f'Optimizasyon ({opt_metot}) — Sharpe: {opt.get("sharpe","?")}',yaxis_title='%')
    else:
        f7.add_annotation(text='Optimizasyon için yfinance verisi gerekli',x=0.5,y=0.5,showarrow=False,font=dict(color=MUTED,size=13))
        f7.update_layout(**PL,title='Portföy Optimizasyonu')

    return html.Div([
        # Metrik kartlar
        html.Div([
            krt(f'Parametrik VaR {ci}%·{hz}G',ftl(R['d_var']),f'{vp:.2f}% portföy · {rl}',vc,'red'),
            krt('CVaR / Beklenen Kayıp',ftl(R['cvar']),'VaR aşımı ort. kaybı',WARNING,'yellow'),
            krt('Monte Carlo VaR',ftl(R['mc_var']),f"{mc_n:,} simülasyon",ACCENT,'blue'),
            krt('Sharpe Oranı',f"{R['sharpe']:.3f}",f"Vol:%{R['yillik_vol']:.1f} G:%{R['yillik_mu']:.1f}",SUCCESS,'green'),
            krt('Portföy Beta',f"{R['p_beta']:.3f}",f"μ={mu*100:.4f}% σ={sig*100:.4f}%",PURPLE,'purple'),
        ],className='metrics-grid'),

        html.Div('⚠️ Portföy yüksek risk taşıyor.',className='alert alert-danger') if vp>5 else html.Div(),

        # Sekmeler
        html.Div([dcc.Tabs(id='tabs',value='t1',className='custom-tabs',children=[
            dcc.Tab(label='📊 Dağılım',     value='t1',className='tab',selected_className='tab-active'),
            dcc.Tab(label='🎲 Monte Carlo', value='t2',className='tab',selected_className='tab-active'),
            dcc.Tab(label='💼 Portföy',     value='t3',className='tab',selected_className='tab-active'),
            dcc.Tab(label='📈 Fiyat',       value='t4',className='tab',selected_className='tab-active'),
            dcc.Tab(label='🔗 Korelasyon',  value='t5',className='tab',selected_className='tab-active'),
            dcc.Tab(label='📉 GARCH',       value='t6',className='tab',selected_className='tab-active'),
            dcc.Tab(label='⚡ Optimizasyon',value='t7',className='tab',selected_className='tab-active'),
            dcc.Tab(label='🔮 Senaryo',     value='t8',className='tab',selected_className='tab-active'),
            dcc.Tab(label='📋 Detay',       value='t9',className='tab',selected_className='tab-active'),
        ]),
        html.Div(id='tab-ic'),
        dcc.Store(id='figs',data={
            't1':f1.to_json(),'t2':f2.to_json(),'t3a':f3a.to_json(),
            't3b':f3b.to_json(),'t4':f4.to_json(),'t5':f5.to_json(),
            't6':f6.to_json(),'t7':f7.to_json(),
            'sen':sen,'katki':R['katki'],
            'p':{'z':R['z'],'p_mu':mu,'p_sig':sig,'vp':vp},
            'mc_meta':{'v':R['mc_var'],'k':R['mc_kotu'],'m':R['mc_medyan']},
            'garch':garch,'opt':opt,
            'guven':guven,'hz':hz,'mc_n':mc_n,
        }),
        ],className='panel',style={'padding':'1.2rem'}),
    ])

# ── Tab içeriği ───────────────────────────────────────────────
@app.callback(Output('tab-ic','children'),Input('tabs','value'),Input('figs','data'))
def tab_ic(t,figs):
    if not figs: return html.Div()
    import plotly.io as pio
    def G(k,h=320): return dcc.Graph(figure=pio.from_json(figs[k]),
                                      config={'displayModeBar':True,'displaylogo':False},
                                      style={'height':f'{h}px'})
    p=figs.get('p',{}); mm=figs.get('mc_meta',{})
    gh=figs.get('garch',{}); op=figs.get('opt',{})
    guven=figs.get('guven',0.99); hz=figs.get('hz',1); mc_n=figs.get('mc_n',10000)
    sen=figs.get('sen',{})

    if t=='t1': return html.Div([html.Div(f"z={p.get('z',0):.4f}  μ={p.get('p_mu',0)*100:.4f}%  σ={p.get('p_sig',0)*100:.4f}%  |  VaR=W×(μ−z·σ)×√T",className='alert alert-info'),G('t1')])
    if t=='t2': return html.Div([html.Div(f"{mc_n:,} senaryo  En kötü: {ftl(mm.get('k',0))}  Medyan: {ftl(mm.get('m',0))}",className='alert alert-info'),G('t2')])
    if t=='t3': return html.Div([html.Div([html.Div(G('t3a',260),style={'flex':'1'}),html.Div(G('t3b',260),style={'flex':'1'})],style={'display':'flex','gap':'1rem'})])
    if t=='t4': return G('t4')
    if t=='t5': return html.Div([html.Div('Gerçek getiri serisinden hesaplanan korelasyon matrisi',className='alert alert-info'),G('t5')])
    if t=='t6':
        if gh.get('dates') and len(gh.get('dates',[])) > 0:
            info = (f"ω={gh.get('omega',0):.6f}  "
                    f"α={gh.get('alpha',0):.4f}  "
                    f"β={gh.get('beta_g',0):.4f}  "
                    f"Tahmin Vol: %{gh.get('tahmin',0)*100:.3f}")
        elif gh.get('hata'):
            info = f"GARCH Hatası: {gh['hata'][:80]}"
        else:
            info = 'GARCH: yfinance verisi bekleniyor...'
        return html.Div([html.Div(info,className='alert alert-info'),G('t6')])
    if t=='t7':
        info=(f"Getiri: %{op.get('ret','?')}  Vol: %{op.get('vol','?')}  Sharpe: {op.get('sharpe','?')}") if op.get('w') else 'Optimizasyon için yfinance verisi gerekli'
        return html.Div([html.Div(info,className='alert alert-info'),G('t7')])
    if t=='t8':
        def sc(title,val,color,extra=''):
            return html.Div([html.Div(title,className='scenario-title',style={'color':color}),
                             html.Div(ftl(val),className='scenario-val',style={'color':color}),
                             html.Div(extra,style={'fontSize':'10px','color':MUTED,'marginTop':'3px','fontFamily':'IBM Plex Mono'})],
                            className='scenario-card')
        return html.Div([html.Div(f'{hz} günlük stres test senaryoları',className='alert alert-warning'),
                         html.Div([sc('🐂 Boğa (+2σ)',sen.get('boga',0),SUCCESS),
                                   sc('📊 Baz',sen.get('baz',0),ACCENT),
                                   sc('🐻 Ayı (−2σ)',sen.get('ayi',0),WARNING),
                                   sc('💥 Kriz (−3.5σ)',sen.get('kriz',0),DANGER,f"Kayıp: {ftl(abs(sen.get('kriz',0)))}")],
                                  className='scenario-grid')])
    if t=='t9':
        rows=figs.get('katki',[])
        cols=[{'name':c,'id':c} for c in (rows[0].keys() if rows else [])]
        return html.Div([dash_table.DataTable(data=rows,columns=cols,
            style_table={'overflowX':'auto'},
            style_cell={'backgroundColor':CARD,'color':TEXT,'border':f'1px solid {GRID}','fontFamily':'IBM Plex Mono','fontSize':'11px','padding':'7px 10px'},
            style_header={'backgroundColor':PANEL,'color':MUTED,'fontWeight':'500','textTransform':'uppercase'},
            style_data_conditional=[{'if':{'filter_query':'{Kaynak} contains "Yahoo"','column_id':'Kaynak'},'color':SUCCESS},
                                     {'if':{'filter_query':'{Kaynak} contains "Fallback"','column_id':'Kaynak'},'color':WARNING}],
            page_size=15),
            html.Div(f'VaR=W×(μ−z·σ)×√T | CVaR=W×σ×φ(z)/(1−α)−W×μ | σₚ²=Σwᵢwⱼσᵢσⱼρᵢⱼ | Sharpe=(μ×252−rᶠ)/(σ×√252) | α={guven*100:.0f}%',
                     className='formula-box')])
    return html.Div()

if __name__ == '__main__':
    app.run_server(debug=False)
