# 🐺 **SAINTv2 — PPO Reinforcement Trading Bot for BTCUSD M1**

### *Backtest • Entraînement PPO • Trading Live MetaTrader 5*

**SAINTv2 (“Loup Ω”)** est un agent de trading basé sur **PPO + SAINT (Self-Attention Across Interleaved Time-series)**.
Il est conçu pour le **scalping BTCUSD en M1**, avec fusion **M1 + H1**, gestion avancée du risque, SL/TP dynamiques basés ATR, break-even intelligent et trailing adaptatif.

Ce dépôt contient :

* ⚡ **Backtests complets** (standard + stress test)
* 🤖 **Exécution live** sur MetaTrader 5
* 🧠 **Entraînement PPO + architecture SAINTv2**
* 📊 Normalisation complète des OHLC + indicateurs
* 🧩 **Modèles pré-entraînés long & short**

---

# 📁 **Contenu du projet**

### 🧪 **Backtests**

Scripts :

* `backtest_saintv2.py`
* `backtest_saintv2_stress_test.py`

Fonctionnalités :

* Fusion M1/H1 via `merge_asof`
* Indicateurs identiques au training :

  * RSI14, ATR14, vol20, returns, range_norm
  * Momentum filter : `mom_5`, `rsi_ok`, `high_vol_regime`
* Moteur de trading :

  * BUY1 / SELL1 / BUY1.8 / SELL1.8 / HOLD
  * SL dynamiques (ATR multipliers)
  * Break-even automatique
  * Trailing intelligent basé ATR
  * Action mask identique au training
* Simulation microstructure :

  * Spread + slippage aléatoire
* Observations `(25 × 20)` identiques au modèle
* **Capital initial = 1000$**
* **Volume fixe = 0.01 lots**

---

### 📡 **Trading Live MetaTrader 5**

Script : `ia_live_mt5.py`

Contenu :

* Récupération live des séries M1/H1 MT5
* Normalisation cohérente avec `norm_stats_ohlc_indics.npz`
* Modèles long + short appelés en parallèle
* Action mask live (long only / short only / duel)
* SL/TP dynamiques dès l’ouverture de la position
* Break-even & trailing en conditions réelles
* Risk scale ajustable

---

### 🧠 **Entraînement PPO + SAINTv2**

Script : `training.py`

L’entraînement est géré via :

* PPO complet :

  * GAE(λ)
  * Clipping
  * Entropy bonus
  * KL target adaptatif
* Environnement RL spécialisé :

  * Observation normalisée M1/H1
  * Embedding état de position :

    * pos, entry_price_scaled, current_price_scaled, risk_scale_history
  * Reward shaping :

    * momentum reward
    * holding penalty
    * latent PnL
    * TP / SL incentives
* Walk-forward training :

  * Split train / validation / test
  * Plusieurs folds
* Modèles générés :

  * `saintv2_loup_long_*`
  * `saintv2_loup_short_*`

#### ⚙️ **Comment entraîner long ou short**

Le script entraîne **automatiquement LONG puis SHORT** :

```python
if __name__ == "__main__":
    cfg_base = PPOConfig()

    # LONG
    cfg_long = PPOConfig(**cfg_base.__dict__)
    cfg_long.side = "long"
    cfg_long.model_prefix = "saintv2_loup_long"
    run_walkforward(cfg_long, train_frac=0.6, val_frac=0.2, test_frac=0.2, max_folds=3)

    # SHORT
    cfg_short = PPOConfig(**cfg_base.__dict__)
    cfg_short.side = "short"
    cfg_short.model_prefix = "saintv2_loup_short"
    run_walkforward(cfg_short, train_frac=0.6, val_frac=0.2, test_frac=0.2, max_folds=3)
```

### ✔️ Entraîner **seulement LONG**

Commenter le bloc short :

```python
# SHORT désactivé
```

Puis lance :

```bash
python training.py
```

### ✔️ Entraîner **seulement SHORT**

Commenter le bloc long :

```python
# LONG désactivé
```

Puis lance :

```bash
python training.py
```

---

# 📊 **Normalisation**

Fichier : `norm_stats_ohlc_indics.npz`

Il contient :

* moyennes
* écarts-types

pour **toutes les features M1/H1**.

⚠️ Obligatoire :
**Training, backtest et live doivent utiliser exactement ces statistiques.**

---

# 🤖 **Modèles pré-entraînés**

Inclus :

* `bestprofit_saintv2_loup_long_wf1_long_wf1.pth`
* `bestprofit_saintv2_loup_short_wf1_short_wf1.pth`

Prêts pour :

* backtest
* live trading
* fine-tuning

---

# 🧪 **Résultats Backtest (capital initial : 1000$, volume 0.01) du 01/12/2024 au 04/12/2025**

## ⭐ Backtest standard

```
===================== RÉSULTATS BACKTEST =====================
Mode side           : duel
Capital initial     : 1000.00
Capital final       : 21933.61
PnL total           : 20933.61
Nb trades           : 14026
Winrate             : 47.4%
PnL moyen / trade   : 1.49
Meilleur trade      : 84.18
Pire trade          : -122.31
Max drawdown (equity): 11.2%
==============================================================
```

## 🔥 Backtest avec Stress Test

```
===================== RÉSULTATS BACKTEST =====================
Mode side           : duel
Capital initial     : 1000.00
Capital final       : 16254.77
PnL total           : 15254.77
Nb trades           : 13906
Winrate             : 46.8%
PnL moyen / trade   : 1.10
Meilleur trade      : 106.14
Pire trade          : -115.59
Max drawdown (equity): 14.2%
==============================================================
```

---

# 🧠 **Architecture SAINTv2**

### Schéma simplifié

```
                ┌───────────────────────┐
                │  Input (25 × 20)      │
                │  OHLC + indicateurs   │
                └──────────┬────────────┘
                           │ Projection
                ┌──────────▼───────────┐
                │  Linear Embedding    │
                └──────────┬───────────┘
                           │ + Positional Encoding
                ┌──────────▼───────────┐
                │    SAINT Block       │
                │  ┌───────────────┐   │
                │  │ RowAttention  │   │ ← dépendances temporelles
                │  ├───────────────┤   │
                │  │ ColAttention  │   │ ← dépendances entre features
                │  ├───────────────┤   │
                │  │ Gated FFN     │   │
                │  └───────────────┘   │
                └──────────┬───────────┘
                           │
           ┌───────────────▼──────────────┐
           │ Actor Head (5 actions)        │
           │ Critic Head (valeur V)        │
           └───────────────────────────────┘
```

---

# 🧠 **Pourquoi SAINTv2 est supérieur aux CNN/LSTM classiques ?**

## ✔️ **1. Capture mieux la microstructure M1**

* RowAttention → dépendances temporelles longues
* ColumnAttention → relations entre features
  → Le modèle “lit” le marché comme une matrice, pas une série simpliste.

## ✔️ **2. Comprend la volatilité et le momentum**

L’attention pondère automatiquement :

* ATR
* RSI
* retournements rapides
  → Les entrées de trade agressives deviennent plus précises.

## ✔️ **3. Rendu robuste grâce à la symétrie Long/Short**

L’architecture apprend :

* patterns haussiers ↔ baissiers
* divergences rapides
* structures de retournement

## ✔️ **4. Architecture légère → parfaite pour RL**

Contrairement aux Transformers complets :

* SAINT = **beaucoup plus rapide**
* parfait pour PPO (beaucoup d’échantillons)

## ✔️ **5. Meilleure généralisation**

Les tests walk-forward montrent une forte stabilité :

* drawdown faible
* performance quasi identique entre périodes d'entraînement

---

# 🛠 **Installation**

## 🔧 Via Miniconda (recommandé)

### 1. Installer Miniconda

Téléchargement : [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

### 2. Créer un environnement

```bash
conda create -n saint python=3.10
conda activate saint
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

Si tu veux CUDA (si tu es sur GPU NVIDIA) :

```txt
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.5.1+cu124
```

---

# ▶️ **Utilisation**

## 🧪 Backtest standard

```bash
python backtest_saintv2.py
```

## 🔥 Stress Test

```bash
python backtest_saintv2_stress_test.py
```

## 📡 Live MT5

```bash
python ia_live_mt5.py
```

## 🧠 Entraînement complet (long + short)

```bash
python training.py
```

---

# ⚠️ Avertissement

Projet expérimental.
Aucune performance n’est garantie.
Utilisation en réel **à vos risques**.

---

