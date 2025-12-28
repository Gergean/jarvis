# Jarvis Mimari Dokümantasyonu

Bu doküman, Jarvis projesinin nasıl çalıştığını, neden bu şekilde tasarlandığını ve temel kavramları açıklar. Genetik algoritmalar hakkında hiçbir ön bilgi gerektirmez.

## İçindekiler

1. [Proje Geçmişi](#proje-geçmişi)
2. [Genetik Algoritma Nedir?](#genetik-algoritma-nedir)
3. [Jarvis'te GA Nasıl Kullanılıyor?](#jarviste-ga-nasıl-kullanılıyor)
   - [Fitness Fonksiyonu Detayları](#fitness-fonksiyonu-detayları)
4. [Kod Yapısı](#kod-yapısı)
5. [Futures Trading Mantığı](#futures-trading-mantığı)
6. [Paper Trading ve Elites Sistemi](#paper-trading-ve-elites-sistemi)
7. [Kullanım Örnekleri](#kullanım-örnekleri)

---

## Proje Geçmişi

### Eski Sistem (Spot Trading)

Projenin ilk versiyonu **spot trading** üzerine kuruluydu:

```
Eski ActionType: BUY, SELL, STAY, ERR
```

- Sadece "al ve sat" yapabiliyorduk
- Fiyat düştüğünde zarar ediyorduk (short yapamıyoruz)
- Sabit sinyal üreticileri vardı: SuperTrend, VWMA, SMA
- Bu sinyal üreticileri hardcoded parametrelerle çalışıyordu

**Problem:** Piyasa düşerken para kazanamıyorduk. ETH %14 getiri sağladı ama aynı dönemde faiz bile daha iyi olabilirdi.

### Yeni Sistem (Futures + GA)

Aralık 2025'te sistemi tamamen yeniden tasarladık:

```
Yeni ActionType: LONG, SHORT, CLOSE, STAY, ERR
```

- **Futures trading**: Long ve short pozisyon açabiliyoruz
- **Genetic Algorithm**: Strateji parametreleri evrimleşiyor
- **Leverage**: 1x-10x ayarlanabilir kaldıraç
- **Funding fee**: Gerçekçi backtest için simüle ediliyor
- **Liquidation**: Kaldıraçlı pozisyonlarda tasfiye simülasyonu

---

## Genetik Algoritma Nedir?

Genetik Algoritma (GA), doğadaki evrim sürecini taklit eden bir optimizasyon tekniğidir. Biyolojiden esinlenen kavramları kullanır.

### Biyoloji Analojisi

| Biyoloji | GA Karşılığı | Jarvis'te |
|----------|--------------|-----------|
| Birey (Organizma) | Individual | Bir trading stratejisi |
| Gen | Parametre | RSI periyodu, SMA değeri, ağırlık |
| Kromozom | Rule seti | Stratejideki tüm kurallar |
| Popülasyon | Population | 50 farklı strateji |
| Uygunluk (Fitness) | Fitness score | Stratejinin getirisi |
| Doğal seçilim | Selection | En karlı stratejileri seç |
| Çaprazlama | Crossover | İki stratejiyi birleştir |
| Mutasyon | Mutation | Rastgele değişiklik yap |
| Nesil | Generation | Bir evrim döngüsü |

### Evrim Nasıl Çalışır?

Doğada evrim şöyle işler:

1. **Varyasyon**: Bireyler birbirinden farklıdır (genler farklı)
2. **Seçilim**: Ortama daha uygun olanlar hayatta kalır
3. **Kalıtım**: Hayatta kalanlar özelliklerini yavruya aktarır
4. **Zaman**: Nesiller boyunca popülasyon iyileşir

GA da aynı mantıkla çalışır:

```
┌─────────────────────────────────────────────────────────────┐
│  BAŞLANGIÇ: 50 rastgele strateji oluştur                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  DEĞERLENDİR: Her stratejiyi backtest yap, fitness hesapla  │
│                                                             │
│  Strateji A: %15 getiri → fitness = 15                      │
│  Strateji B: %3 getiri  → fitness = 3                       │
│  Strateji C: %-5 getiri → fitness = -5                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  SEÇ: En iyi %10'u koru (elitism)                           │
│                                                             │
│  Strateji A ✓ (elit)                                        │
│  Strateji B ✓ (elit)                                        │
│  Strateji C ✗ (elenecek)                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ÇAPRAZLA: İki iyi stratejiyi birleştirerek yeni oluştur    │
│                                                             │
│  Anne: RSI(14) > 70, SMA(50) > 90000                        │
│  Baba: RSI(21) > 65, EMA(20) > 85000                        │
│  Çocuk: RSI(14) > 70, EMA(20) > 85000  ← karışım            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  MUTASYON: Rastgele küçük değişiklikler yap (%10 şans)      │
│                                                             │
│  RSI(14) > 70  →  RSI(14) > 72  (target değişti)            │
│  SMA(50) > 90000  →  EMA(50) > 90000  (indicator değişti)   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  YENİ NESİL: 50 strateji (%10 elit + %10 rastgele + çocuklar)│
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    30 nesil tekrarla
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  SONUÇ: En iyi stratejiyi kaydet                            │
└─────────────────────────────────────────────────────────────┘
```

### Neden GA Kullanıyoruz?

**Problem:** Trading stratejisi oluşturmak için binlerce parametre kombinasyonu var.

Örnek bir strateji düşün:
- RSI periyodu: 5-50 arası (45 seçenek)
- RSI eşik değeri: 20-80 arası (60 seçenek)
- SMA periyodu: 10-200 arası (190 seçenek)
- Ağırlıklar: -1 ile +1 arası (sonsuz seçenek)

Tüm kombinasyonları denemek → 45 × 60 × 190 × ... = **milyonlarca** olasılık

**Çözüm:** GA, tüm olasılıkları denemek yerine, iyi olanları evrimleştirerek hızlıca optimum noktaya yaklaşır.

```
Brute Force: ████████████████████████████████ 1,000,000 deneme
GA:          ████ 1,500 deneme (50 birey × 30 nesil)

Sonuç: GA, %0.3 çabayla %90+ performansa ulaşabilir
```

---

## Jarvis'te GA Nasıl Kullanılıyor?

### Individual (Birey) = Strateji

Bir Individual, trading kararı veren bir stratejidir. İçinde kurallar (rules) barındırır.

```python
class Individual:
    rules: list[Rule]      # Kurallar listesi
    fitness: float = 0.0   # Ne kadar karlı?
```

**Örnek bir strateji:**
```
BTCUSDT Stratejisi (fitness: +12.5):
├── Kural 1: RSI(14) > 70 ise → +0.8 puan
├── Kural 2: SMA(50) > 95000 ise → -0.5 puan
├── Kural 3: MACD_HIST > 0 ise → +0.3 puan
└── Kural 4: Hacim > 1B ise → +0.2 puan
```

### Rule (Kural) = Tek Bir Koşul

Her kural, bir göstergenin değerini hedefle karşılaştırarak katkı hesaplar.

```python
class Rule:
    indicator: Indicator  # RSI, SMA, MACD, vs.
    target: float         # Hedef değer (örn: 70)
    weight: float         # Ağırlık (±1,000,000 arası)
```

**Kural nasıl çalışır?**

```
katkı = (gösterge_değeri - hedef) * ağırlık / 100,000
```

Örnek:
```
RSI = 75, hedef = 70, ağırlık = +500,000

katkı = (75 - 70) * 500,000 / 100,000
      = 5 * 5
      = +25 (güçlü pozitif katkı)
```

Hedeften uzaklık ve ağırlık birlikte sinyalin gücünü belirler.

### Sinyal Nasıl Üretilir?

Tüm kuralların katkıları toplanır ve eşik değerlere göre karar verilir:

```
toplam = Σ (gösterge_değeri - hedef) * ağırlık / 100,000
```

Karar mantığı:
```
Pozisyon yok iken:
  toplam > +1.0  → LONG aç
  toplam < -1.0  → SHORT aç

Long pozisyondayken:
  toplam < -0.5  → CLOSE (trend dönüyor)

Short pozisyondayken:
  toplam > +0.5  → CLOSE (trend dönüyor)

Aksi halde → STAY (bekle)
```

**Örnek hesaplama:**

```
Kural 1: RSI(14)=75, hedef=70, ağırlık=+100,000 → (75-70)*100,000/100,000 = +5.0
Kural 2: SMA(50)=92000, hedef=95000, ağırlık=+50,000 → (92000-95000)*50,000/100,000 = -1.5
Kural 3: MACD_HIST=150, hedef=0, ağırlık=+20,000 → (150-0)*20,000/100,000 = +30.0
                                                                            ─────
                                                                 Toplam:   +33.5

33.5 > 1.0 → LONG aç!
```

### Population (Popülasyon) = Strateji Havuzu

100 farklı strateji aynı anda yarışır (train komutu varsayılanı):

```python
class Population:
    individuals: list[Individual]  # 100 strateji (train varsayılanı)
    generation: int = 0            # Şu anki nesil

    # Ayarlar (Population sınıfı varsayılanı 50, train komutu 100 kullanır)
    population_size: int = 100     # Kaç birey?
    elitism_ratio: float = 0.1     # En iyi %10'u koru (10 birey)
    mutation_rate: float = 0.1     # %10 mutasyon şansı
```

### Fitness (Uygunluk) = Başarı Ölçüsü

Bir stratejinin ne kadar iyi olduğunu ölçer. İki farklı hesaplama yöntemi vardır:

**1. Walk-Forward Validation (Varsayılan - train komutu):**

Walk-forward modunda strateji birden fazla pencerede (window) test edilir:

```
fitness = Σ(pencere_getirisi) - Σ(pencere_drawdown)
```

- Tüm pencerelerin getirisi toplanır
- Tüm pencerelerin maksimum drawdown'ı toplanır
- Herhangi bir pencerede liquidation olursa → fitness = 0

Bu formül hem getiriyi maksimize etmeyi hem de riski minimize etmeyi teşvik eder.

**2. Tek Dönem Değerlendirme (Population.evaluate_fitness):**

```
fitness = strateji_getirisi - buy_hold_getirisi
```

**Walk-Forward Fitness Örnekleri:**

| Pencere 1 | Pencere 2 | Pencere 3 | DD Toplamı | Fitness |
|-----------|-----------|-----------|------------|---------|
| +5% | +3% | +2% | 6% | (10) - 6 = 4 |
| +10% | -2% | +5% | 8% | (13) - 8 = 5 |
| +15% | LIQ | +3% | - | 0 (liquidation!) |

### Fitness Fonksiyonu Detayları

Fitness hesaplaması `train.py` modülünde walk-forward validation ile yapılır. Her strateji için tüm pencerelerde gerçekçi futures backtest simülasyonu çalıştırılır.

#### Walk-Forward Formülü

```
fitness = Σ(window_return_pct) - Σ(window_max_drawdown_pct)

Herhangi bir pencerede liquidation varsa:
  fitness = 0
```

Pozitif fitness = Strateji, tüm pencerelerde tutarlı performans gösterdi.

#### Simülasyon Kuralları

**Pozisyon Açma:**
- Her pozisyon, mevcut bakiyenin %20'sini kullanır (investment_ratio)
- Kaldıraç uygulanır: pozisyon_büyüklüğü = margin × leverage
- Açılışta taker fee (%0.04) kesilir

**Pozisyon Kapama:**
- Kar/zarar hesaplanır ve bakiyeye eklenir
- Kapanışta da taker fee kesilir
- Açık pozisyon varsa dönem sonunda unrealized PnL dahil edilir

**Liquidation (Tasfiye):**
- Kaldıraçlı pozisyonda fiyat tasfiye noktasına ulaşırsa margin kaybedilir
- Örnek: 10x long'da fiyat %10 düşerse → tüm margin kayıp
- Tasfiye olan stratejiler çok düşük fitness alır

**Funding Fee:**
- Her 8 saatte bir uygulanır
- Long pozisyonlar öder, short pozisyonlar alır
- Uzun süre açık kalan pozisyonları cezalandırır

#### Neden Bu Yöntem?

| Özellik | Açıklama |
|---------|----------|
| **Walk-Forward Validation** | Birden fazla pencerede test ederek overfitting önlenir |
| **Drawdown Cezası** | Yüksek drawdown, yüksek getiriyi dengeler |
| **Liquidation = 0** | Tek bir liquidation bile stratejiyi diskalifiye eder |
| **Gerçekçi maliyetler** | Komisyon, funding, liquidation simüle edilir |
| **Risk yönetimi** | %20 investment ratio ile tek pozisyon tüm sermayeyi riske atmaz |
| **Position awareness** | LONG'dayken tekrar LONG açılmaz, önce CLOSE gerekir |

> **TODO:** Sharpe Ratio'yu fitness fonksiyonuna ekleyerek risk-adjusted getiri ölçümü denenecek.

### Crossover (Çaprazlama) = İki Stratejiyi Birleştirme

İki başarılı strateji "ebeveyn" olur, kuralları karıştırılarak "çocuk" oluşturulur:

```
Anne Strateji:                    Baba Strateji:
├── RSI(14) > 70 * +0.8          ├── RSI(21) > 65 * +0.6
├── SMA(50) > 95000 * -0.5       ├── EMA(20) > 90000 * +0.4
└── MACD > 0 * +0.3              └── VOLUME(20) > 1.5 * -0.2

Çocuk (rastgele seçim):
├── RSI(14) > 70 * +0.8    ← Anneden
├── EMA(20) > 90000 * +0.4 ← Babadan
└── MACD > 0 * +0.3        ← Anneden
```

**Neden çaprazlama?**

Her iki ebeveyn de başarılı. Belki annenin RSI kuralı çok iyi, babanın EMA kuralı çok iyi. İkisini birleştirince her ikisinin de iyi özelliklerini alan bir çocuk elde edebiliriz.

### Mutation (Mutasyon) = Rastgele Değişiklik

Bazen kurallar rastgele değiştirilir. Bu, yeni olasılıkların keşfedilmesini sağlar:

```
Orijinal: RSI(14) > 70 * +0.8

Mutasyon tipleri:
1. Indicator değişir: RSI(14) → RSI(21) veya EMA(14)
2. Target değişir:    70 → 65 veya 75
3. Weight değişir:    +0.8 → +0.6 veya +0.9
```

**Neden mutasyon?**

Çaprazlama sadece mevcut genleri karıştırır. Eğer hiçbir stratejide EMA(200) yoksa, çaprazlamayla asla EMA(200) elde edemeyiz. Mutasyon, yeni genlerin ortaya çıkmasını sağlar.

### Tournament Selection = Ebeveyn Seçimi

Ebeveyn seçmek için "turnuva" yapılır:

```
Havuzdan rastgele 3 strateji seç:
├── Strateji A: fitness = 12
├── Strateji B: fitness = 5
└── Strateji C: fitness = -2

En yüksek fitness'a sahip olan kazanır → Strateji A ebeveyn olur
```

Bu yöntem, iyi stratejilere daha fazla şans verir ama kötü olanlara da küçük bir şans tanır (çeşitlilik için).

---

## Kod Yapısı

### Dosya Haritası

```
jarvis/
│
├── src/jarvis/            # Ana kaynak kodu
│   ├── commands/          # CLI komutları
│   │   ├── train.py       # GA eğitimi (walk-forward validation)
│   │   ├── test.py        # Strateji testi
│   │   ├── trade.py       # Canlı trading
│   │   ├── paper.py       # Paper trading simülasyonu
│   │   ├── download.py    # Veri indirme
│   │   └── pinescript.py  # TradingView export
│   │
│   ├── genetics/          # Genetik algoritma çekirdeği
│   │   ├── individual.py  # Individual sınıfı (strateji)
│   │   ├── population.py  # Population sınıfı (evrim motoru)
│   │   ├── rule.py        # Rule sınıfı (tek kural)
│   │   ├── indicators.py  # RSI, SMA, MACD hesaplamaları
│   │   └── strategy.py    # Strateji kaydetme/yükleme
│   │
│   ├── client.py          # Binance API + FakeClient
│   ├── models.py          # ActionType, PositionSide, vs.
│   ├── settings.py        # Ayarlar (.env dosyasından)
│   ├── utils.py           # Yardımcı fonksiyonlar
│   └── logging.py         # Log ayarları (console only)
│
├── data/binance/          # Tarihsel veriler
│   └── {SYMBOL}/{interval}/YYYYMMDD.csv
│
├── strategies/            # Stratejiler
│   ├── *.json             # Eğitilmiş stratejiler
│   ├── *.pine             # TradingView Pine Script
│   └── elites/            # Günlük evrimleşen elite'ler
│       └── {SYMBOL}/{interval}/YYYYMMDD_HHMMSS.json
│
├── paper/                 # Paper trading wallet'ları
│   └── {wallet_id}.json
│
└── results/               # Test sonuçları
    └── {strategy_id}_{interval}_{dates}.json
```

### Veri Akışı

```
┌─────────────────────────────────────────────────────────────┐
│                      just download                           │
│  Binance API → CSV dosyaları                                │
│  data/binance/BTCUSDT/1h/20240101.csv                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       just train                             │
│                                                             │
│  1. CSV'leri oku → OHLCV verisi                             │
│  2. 100 rastgele strateji oluştur                           │
│  3. 30 nesil evrimleştir                                    │
│  4. En iyi stratejiyi kaydet                                │
│     strategies/BTCUSDT_abc123.json                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                        just test                             │
│                                                             │
│  1. Stratejiyi yükle                                        │
│  2. FARKLI zaman diliminde backtest yap                     │
│  3. Sonuçları karşılaştır (overfitting kontrolü)            │
│     results/BTCUSDT_abc123_test.json                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       just trade                             │
│                                                             │
│  1. Stratejiyi yükle                                        │
│  2. Her saatte:                                             │
│     a. Binance'den son verileri al                          │
│     b. Sinyal hesapla                                       │
│     c. Gerekirse pozisyon aç/kapat                          │
│  3. Telegram'a bildirim gönder                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Futures Trading Mantığı

### Spot vs Futures

**Spot Trading (Eski sistem):**
```
BTC = $100
AL → BTC = $120 → SAT → %20 kar ✓
AL → BTC = $80  → SAT → %20 zarar ✗

Sadece fiyat yükselirse para kazanabilirsin.
```

**Futures Trading (Yeni sistem):**
```
LONG (fiyat artacak diye bahis):
BTC = $100, LONG aç → BTC = $120 → Kapat → %20 kar ✓

SHORT (fiyat düşecek diye bahis):
BTC = $100, SHORT aç → BTC = $80 → Kapat → %20 kar ✓

Her iki yönde de para kazanabilirsin!
```

### Pozisyon Durumları

```
PositionSide.NONE   → Pozisyon yok, bekliyoruz
PositionSide.LONG   → Uzun pozisyon açık (fiyat artarsa kar)
PositionSide.SHORT  → Kısa pozisyon açık (fiyat düşerse kar)
```

### Leverage (Kaldıraç)

Kaldıraç, az parayla büyük pozisyon açmayı sağlar:

```
Sermaye: $100
Kaldıraç: 10x
Pozisyon büyüklüğü: $100 × 10 = $1000

BTC %5 yükselirse:
- Kaldıraçsız: $100 × 5% = $5 kar
- 10x kaldıraç: $1000 × 5% = $50 kar (%50 getiri!)

AMA

BTC %10 düşerse:
- Kaldıraçsız: $100 × 10% = $10 zarar
- 10x kaldıraç: $1000 × 10% = $100 zarar (TÜM SERMAYENİ KAYBETTİN!)
```

### Liquidation (Tasfiye)

Kaldıraçlı pozisyonda, belirli bir zararda pozisyon zorla kapatılır:

```
10x LONG pozisyon, giriş fiyatı: $100

Tasfiye fiyatı = giriş × (1 - 1/kaldıraç)
               = $100 × (1 - 1/10)
               = $100 × 0.9
               = $90

BTC $90'a düşerse → POZİSYON TASFİYE, $100 margin kayıp!
```

### Funding Fee

Futures piyasasında her 8 saatte bir ücret alınır/ödenir:

```
FUNDING_FEE_RATE = %0.01 (her 8 saatte)

LONG pozisyondaysan: Fee ÖDERSIN (genelde)
SHORT pozisyondaysan: Fee ALIRSIN (genelde)

Örnek:
$10,000 pozisyon × %0.01 = $1 her 8 saatte
Günde 3 kez = $3/gün
Ayda = ~$90
```

Bu backtest'te simüle edilir, gerçekçi sonuçlar için önemlidir.

---

## Paper Trading ve Elites Sistemi

### Paper Trading Nedir?

Paper trading, gerçek para kullanmadan simülasyon ortamında trade yapmaktır. Stratejiyi canlıya almadan önce test etmek için kullanılır.

```bash
# Wallet oluştur (seed strateji gerekli)
uv run python src/jarvis.py paper init test1 -b 1000 -c ETHUSDT:1h -s ETHUSDT_abc123

# Belirli tarihe kadar trade simüle et
uv run python src/jarvis.py paper trade test1 -et 2025-10-15T00:00:00

# Wallet durumunu gör
uv run python src/jarvis.py paper info test1
```

### Elites Sistemi

Paper trading sırasında "time travel" problemini önlemek için elites sistemi kullanılır.

**Problem:** 70. günde paper trade yaparken, 100. günde (bugün) eğitilmiş stratejiyi kullanırsak, geleceği bilmiş oluruz. Bu gerçekçi değil.

**Çözüm:** Her gün 00:00 UTC'de yeni bir "elite" strateji evrimleştirilir ve kaydedilir:

```
strategies/elites/
└── ETHUSDT/
    └── 1h/
        ├── 20251001_000000.json  # 1 Ekim'de evrimleşen
        ├── 20251002_000000.json  # 2 Ekim'de evrimleşen
        ├── 20251003_000000.json  # 3 Ekim'de evrimleşen
        └── ...
```

Paper trade şöyle çalışır:
1. Simülasyon tarihi 00:00 UTC ise → Yeni elite evolve et ve kaydet
2. Değilse → O tarihte mevcut olan en son elite'i kullan
3. Elite yoksa → Seed stratejiyi kullan

Bu sayede:
- Her gün farklı bir strateji kullanılır
- Gelecek bilgisi kullanılmaz
- Gerçek canlı trading ile aynı koşullar simüle edilir

### Seed Strateji

Paper trading başlatırken bir "seed" strateji gereklidir. Bu, evrim zincirinin başlangıç noktasıdır:

```bash
# ETHUSDT_abc123 stratejisi seed olarak kullanılır
uv run python src/jarvis.py paper init mywallet -b 1000 -c ETHUSDT:1h -s ETHUSDT_abc123
```

Evrim süreci:
```
Seed (ETHUSDT_abc123)
    ↓ evolve (30 gün veri, 10 generation)
Elite Day 1
    ↓ evolve
Elite Day 2
    ↓ evolve
Elite Day 3
    ...
```

Her elite, bir öncekinden evrimleşir. Bu sayede strateji piyasa koşullarına adapte olur.

---

## Kullanım Örnekleri

### Temel Komutlar

```bash
# Veri indir (son 1 yıl)
uv run python src/jarvis.py download -s BTCUSDT ETHUSDT -i 1h

# Strateji eğit (walk-forward validation varsayılan)
uv run python src/jarvis.py train -s BTCUSDT -i 1h

# 5x kaldıraçla eğit
uv run python src/jarvis.py train -s BTCUSDT -i 1h -l 5

# Özel walk-forward periyotlarıyla eğit
uv run python src/jarvis.py train -s BTCUSDT -i 1h --train-period 90d --test-period 2w --step-period 1w

# Walk-forward olmadan eğit (önerilmez)
uv run python src/jarvis.py train -s BTCUSDT -i 1h --no-walk-forward

# Stratejiyi test et
uv run python src/jarvis.py test -s BTCUSDT_abc123 -i 1h

# Simülasyon modunda trade
uv run python src/jarvis.py trade-ga -s BTCUSDT_abc123 --dry-run

# Paper trading
uv run python src/jarvis.py paper init test1 -b 1000 -c BTCUSDT:1h -s BTCUSDT_abc123
uv run python src/jarvis.py paper trade test1 -et 2025-10-15T00:00:00
uv run python src/jarvis.py paper info test1

# Pine Script export
uv run python src/jarvis.py pinescript -s BTCUSDT_abc123
```

### Eğitim Çıktısı Nasıl Okunur?

```
Gen 0: best=-2.92, avg=-7.71
  Elites:
    #1 fitness=-2.92 rules=8
    #2 fitness=-2.92 rules=8
    #3 fitness=-2.92 rules=8
```

- `Gen 0`: İlk nesil (rastgele stratejiler)
- `best=-2.92`: En iyi strateji Buy & Hold'dan %2.92 kötü
- `avg=-7.71`: Ortalama fitness (çoğu strateji kötü başlar)
- `rules=8`: Stratejide 8 kural var

```
Gen 29: best=0.63, avg=-2.08
  Elites:
    #1 fitness=0.63 rules=7
```

- 30 nesil sonra en iyi fitness +0.63
- Bu strateji Buy & Hold'u %0.63 yendi

### Strateji JSON Formatı

```json
{
  "id": "BTCUSDT_abc123",
  "symbol": "BTCUSDT",
  "created_at": "2024-12-23T10:00:00",
  "individual": {
    "rules": [
      {
        "indicator": {"type": "RSI", "period": 14},
        "target": 65.5,
        "weight": 0.8
      },
      {
        "indicator": {"type": "SMA", "period": 50},
        "target": 95000,
        "weight": -0.6
      }
    ]
  },
  "training": {
    "interval": "1h",
    "start_date": "2024-06-01",
    "end_date": "2024-12-01",
    "generations": 30,
    "population_size": 100
  }
}
```

---

## Sık Sorulan Sorular

### Overfitting nedir?

Strateji eğitim verisine "aşırı uyum" sağladığında, yeni verilerde kötü performans gösterir.

```
Eğitim dönemi: %50 getiri 🎉
Test dönemi:   %5 getiri  😢 ← Overfitting!
```

**Çözüm:** Farklı dönemlerde test et. Eğitim ve test sonuçları benzer olmalı.

### Neden bazen fitness negatif?

Negatif fitness = Strateji, "sadece BTC tut" stratejisinden kötü.

Bu normaldir, özellikle ilk nesillerde. Evrim ilerledikçe fitness artmalı.

### Kaç nesil eğitmeli?

- **30 nesil**: Makul sonuçlar (varsayılan)
- **50+ nesil**: Daha iyi sonuçlar ama overfitting riski artar
- **100+ nesil**: Genellikle gereksiz, erken durma yapılabilir

### Hangi interval daha iyi?

- **1h**: Dengeli, çoğu durum için iyi
- **4h**: Daha az trade, daha az komisyon
- **1d**: Uzun vadeli, çok az trade
- **15m/5m**: Çok trade, komisyonlar kar yer

---

## Sonuç

Bu sistem sayesinde:

1. **Otomatik optimizasyon**: Elle parametre aramak yerine evrim bulsun
2. **Her coin'e özel**: BTCUSDT ve TRXUSDT farklı stratejiler kullanabilir
3. **Futures desteği**: Hem yükselen hem düşen piyasadan kar
4. **Gerçekçi backtest**: Funding fee, komisyon, tasfiye simüle edilir
5. **Overfitting kontrolü**: Out-of-sample test ile doğrulama

Sistem sürekli geliştirilebilir:
- Yeni göstergeler eklenebilir
- Fitness fonksiyonu değiştirilebilir (Sharpe ratio?)
- Popülasyon parametreleri ayarlanabilir
