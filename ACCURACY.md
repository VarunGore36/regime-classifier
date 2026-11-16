# Regime classifier accuracy

**Bottom line:** the classifier beats a coin flip at predicting volatility, but it loses to simply looking at the VIX, and it cannot predict direction.

- **Asset:** SPY, daily data
- **Period tested:** Feb 2009 to Oct 2026 (4,457 trading days)
- **Method:** each day's label is checked against what happened over the following 10 sessions
- **Data as of:** 13 Nov 2026 close

## Predicting "the next 10 days will be more volatile than usual"

Half of all days are "more volatile than usual", so a coin flip scores 50%.

| Method | Accuracy | Precision | Recall | Days flagged |
|---|---|---|---|---|
| Classifier: TRANSITION / STRESS / CRISIS | 60.4% | 59.0% | 68.0% | 57.6% |
| Classifier: STRESS / CRISIS only | 63.0% | 70.1% | 45.1% | 32.2% |
| Baseline: VIX above its median | 74.4% | 74.4% | 74.4% | 50.0% |
| Baseline: realized vol (20-day) above its median | 67.7% | 67.7% | 67.7% | 50.0% |

When the classifier says STRESS or CRISIS, high volatility followed 70% of the time. A one-line VIX rule still does better overall.

## Predicting "the next 10 days will be turbulent" (top quarter of volatility)

A quarter of all days qualify, so always guessing "no" scores 75% accuracy; precision and recall matter more here.

| Method | Accuracy | Precision | Recall |
|---|---|---|---|
| Classifier: TRANSITION / STRESS / CRISIS | 56.5% | 34.0% | 78.3% |
| Classifier: STRESS / CRISIS only | 71.8% | 44.9% | 57.8% |
| Baseline: VIX above its median | 68.3% | 43.3% | 86.6% |
| Baseline: realized vol (20-day) above its median | 64.9% | 39.9% | 79.7% |

## Ranking power of each signal (AUC)

0.5 is a coin flip, 1.0 is perfect.

| Signal | High-vol target | Turbulent target | Rank correlation with future vol |
|---|---|---|---|
| Classifier risk score | 0.749 | 0.783 | 0.50 |
| VIX alone | 0.828 | 0.850 | 0.67 |
| Realized vol (20-day) alone | 0.753 | 0.789 | 0.52 |
| Entropy ratio (rho) alone | 0.620 | 0.639 | not measured |

The entropy ratio is the weakest input on its own. The combined risk score is no better than realized vol alone, so the extra signals are currently adding noise rather than information.

## Catching big drops

185 days were followed by a fall worse than -5% within 10 sessions.

- **81%** of them were already flagged TRANSITION or worse.
- **57%** were flagged STRESS or CRISIS.

The classifier flags 58% of all days as TRANSITION or worse, so part of that 81% comes from flagging often.

## Predicting direction

The market rose over the next 10 days 64.8% of the time, so that is the number to beat.

| Signal | Correct | Sample |
|---|---|---|
| CALM_BULL (predicts up) | 63.3% | 1,237 days |
| CALM_BEAR (predicts down) | 45.1% | 182 days |
| Sign of the PCA structure score | 52.8% | all days |

None of these beats always guessing "up". The classifier has no directional skill.

## What followed each label (next 10 sessions)

| Regime | Share of days | Realized vol after | Avg return after | Worst return after | % of periods down |
|---|---|---|---|---|---|
| CALM_BULL | 27.8% | 11.4 | +0.48% | -6.8% | 36.7% |
| CALM_RANGE | 10.6% | 12.0 | +0.21% | -11.6% | 39.8% |
| CALM_BEAR | 4.1% | 12.6 | +0.25% | -9.1% | 45.1% |
| TRANSITION | 25.5% | 13.1 | +0.49% | -17.6% | 33.7% |
| STRESS | 26.0% | 16.5 | +0.72% | -18.2% | 33.3% |
| CRISIS | 6.2% | 33.8 | +1.63% | -26.5% | 28.8% |

## Caveats

- **Overlapping windows:** the 10-day windows overlap, so there are only about 445 independent periods. The real uncertainty is wider than 4,457 days suggests.
- **Cut-offs see the whole sample:** the "above median" lines for the targets and both baselines use the full history, which slightly flatters all methods equally.
- **Untuned weights:** the risk-score weights and the STRESS / TRANSITION cut-offs were first guesses and never fitted.
- **Missing layers:** dealer gamma, strike clusters, flow and dark-pool signals from the guides are not modelled because they need paid data.
- **Daily bars:** the guide's entropy model uses 5-minute NDX bars; this uses daily SPY closes.
- **Not a trading backtest:** nothing here shows that sizing by the classifier makes or saves money.

Educational tool, not investment advice.