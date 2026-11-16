# Regime Classifier - Visual Analysis

This document uses Mermaid diagrams (rendered natively on GitHub) to visualize the accuracy data from [ACCURACY.md](ACCURACY.md).

---

## 1. Predicting "More Volatile Than Usual" (Next 10 Days)

### Accuracy Comparison

```mermaid
bar
    title Accuracy: Predicting High Volatility (Next 10 Days)
    x-axis [Method, Accuracy]
    y-axis "Accuracy %" 0 --> 80
    bar "Classifier (TRANSITION/STRESS/CRISIS)" 60.4
    bar "Classifier (STRESS/CRISIS only)" 63.0
    bar "Baseline: VIX > Median" 74.4
    bar "Baseline: Realized Vol > Median" 67.7
    bar "Coin Flip" 50
```

### Precision, Recall & Days Flagged

```mermaid
graph LR
    subgraph "Classifier: TRANSITION/STRESS/CRISIS"
        A1[Precision: 59.0%] --> A2[Recall: 68.0%] --> A3[Days Flagged: 57.6%]
    end
    subgraph "Classifier: STRESS/CRISIS only"
        B1[Precision: 70.1%] --> B2[Recall: 45.1%] --> B3[Days Flagged: 32.2%]
    end
    subgraph "Baseline: VIX > Median"
        C1[Precision: 74.4%] --> C2[Recall: 74.4%] --> C3[Days Flagged: 50.0%]
    end
    subgraph "Baseline: Realized Vol > Median"
        D1[Precision: 67.7%] --> D2[Recall: 67.7%] --> D3[Days Flagged: 50.0%]
    end
```

### Key Insight Flowchart

```mermaid
flowchart TD
    A[Goal: Predict if next 10 days<br/>more volatile than usual] --> B{Which method?}
    B --> C[Classifier TRANSITION/STRESS/CRISIS]
    B --> D[Classifier STRESS/CRISIS only]
    B --> E[VIX > Median]
    B --> F[Realized Vol > Median]
    C --> G[Accuracy: 60.4%<br/>Precision: 59.0%<br/>Recall: 68.0%]
    D --> H[Accuracy: 63.0%<br/>Precision: 70.1%<br/>Recall: 45.1%]
    E --> I[Accuracy: 74.4% ✓<br/>Precision: 74.4%<br/>Recall: 74.4%]
    F --> J[Accuracy: 67.7%<br/>Precision: 67.7%<br/>Recall: 67.7%]
    G --> K[Beats coin flip ✓<br/>Loses to VIX ✗]
    H --> K
    I --> K
    J --> K
```

---

## 2. Predicting "Turbulent" (Top 25% Volatility)

### Accuracy Comparison

```mermaid
bar
    title Accuracy: Predicting Turbulent Days (Top 25% Volatility)
    x-axis [Method, Accuracy]
    y-axis "Accuracy %" 50 --> 80
    bar "Classifier (TRANSITION/STRESS/CRISIS)" 56.5
    bar "Classifier (STRESS/CRISIS only)" 71.8
    bar "Baseline: VIX > Median" 68.3
    bar "Baseline: Realized Vol > Median" 64.9
    bar "Always Guess No" 75
```

### Precision & Recall (More Important Here)

```mermaid
xychart-beta
    title "Precision vs Recall: Turbulent Target"
    x-axis [TRANSITION/STRESS/CRISIS, STRESS/CRISIS, VIX Median, Realized Vol Median]
    y-axis "Score %" 0 --> 100
    bar [34.0, 44.9, 43.3, 39.9]
    bar [78.3, 57.8, 86.6, 79.7]
    legend ["Precision", "Recall"]
```

### Trade-off Visualization

```mermaid
flowchart LR
    subgraph "High Recall / Low Precision"
        R1[TRANSITION/STRESS/CRISIS<br/>Recall: 78.3%<br/>Precision: 34.0%]
        R2[VIX > Median<br/>Recall: 86.6%<br/>Precision: 43.3%]
    end
    subgraph "Balanced"
        B1[STRESS/CRISIS only<br/>Recall: 57.8%<br/>Precision: 44.9%]
        B2[Realized Vol Median<br/>Recall: 79.7%<br/>Precision: 39.9%]
    end
    R1 -.->|Flags 57.6% of days| R1
    R2 -.->|Flags 50% of days| R2
    B1 -.->|Flags 32.2% of days| B1
    B2 -.->|Flags 50% of days| B2
```

---

## 3. Signal Ranking Power (AUC)

### AUC Comparison

```mermaid
bar
    title AUC: Ranking Power for Volatility Targets
    x-axis [Signal, High-Vol AUC, Turbulent AUC, Rank Corr]
    y-axis "AUC / Correlation" 0 --> 1.0
    bar "Classifier Risk Score" 0.749
    bar "VIX Alone" 0.828
    bar "Realized Vol (20d)" 0.753
    bar "Entropy Ratio (ρ)" 0.620
```

### Detailed AUC Breakdown

```mermaid
xychart-beta
    title "AUC Breakdown by Signal"
    x-axis [High-Vol Target, Turbulent Target, Rank Correlation]
    y-axis "Score" 0 --> 1.0
    line "Classifier Risk Score" [0.749, 0.783, 0.50]
    line "VIX Alone" [0.828, 0.850, 0.67]
    line "Realized Vol (20d)" [0.753, 0.789, 0.52]
    line "Entropy Ratio (ρ)" [0.620, 0.639, null]
```

### Signal Quality Flowchart

```mermaid
flowchart TD
    A[Signal Quality Assessment] --> B{AUC > 0.8?}
    B -->|Yes| C[VIX Alone: 0.828 / 0.850 ✓]
    B -->|No| D{0.7 - 0.8?}
    D -->|Yes| E[Realized Vol: 0.753 / 0.789 ✓]
    D -->|Yes| F[Risk Score: 0.749 / 0.783 ~]
    D -->|No| G[Entropy Ratio: 0.620 / 0.639 ✗]
    C --> H[Best standalone predictor]
    E --> I[Strong baseline]
    F --> J[No better than realized vol]
    G --> K[Weakest input - adds noise]
    J --> L[Extra signals not adding info]
    K --> L
```

---

## 4. Catching Big Drops (> -5% in 10 Days)

```mermaid
pie showData
    title "185 Big Drop Days: Which Regime Flagged Them?"
    "Flagged TRANSITION or worse (81%)" : 150
    "Flagged STRESS or CRISIS (57%)" : 105
    "Not flagged (19%)" : 35
```

### Detection Funnel

```mermaid
flowchart TD
    A[All Trading Days: 4,457] --> B{Big drop next 10 days?}
    B -->|Yes: 185 days| C[TRANSITION or worse: 81% caught]
    B -->|No: 4,272 days| D[Normal days]
    C --> E[STRESS or CRISIS: 57% caught]
    C --> F[Not flagged: 19%]
    E --> G[Classifier flags 58% of ALL days as<br/>TRANSITION or worse → High false positive rate]
```

---

## 5. Predicting Direction

### Directional Accuracy

```mermaid
bar
    title Directional Prediction Accuracy
    x-axis [Signal, Accuracy]
    y-axis "Accuracy %" 0 --> 70
    bar "Market always UP (baseline)" 64.8
    bar "CALM_BULL predicts UP" 63.3
    bar "PCA Structure Score" 52.8
    bar "CALM_BEAR predicts DOWN" 45.1
    bar "Coin Flip" 50
```

### Signal Performance Matrix

```mermaid
graph LR
    subgraph "Market Bias"
        MB[Market UP 64.8% of time]
    end
    subgraph "Classifier Signals"
        S1[CALM_BULL → UP: 63.3% ✗<br/>Worse than baseline]
        S2[CALM_BEAR → DOWN: 45.1% ✗<br/>Worse than coin flip]
        S3[PCA Score Sign: 52.8% ✗<br/>Worse than baseline]
    end
    MB --> S1
    MB --> S2
    MB --> S3
    S1 --> NC[No directional skill]
    S2 --> NC
    S3 --> NC
```

---

## 6. What Followed Each Regime (Next 10 Sessions)

### Realized Volatility After Each Regime

```mermaid
bar
    title Average Realized Volatility (Next 10 Sessions)
    x-axis [Regime, Realized Vol]
    y-axis "Annualized Vol %" 0 --> 40
    bar "CALM_BULL" 11.4
    bar "CALM_RANGE" 12.0
    bar "CALM_BEAR" 12.6
    bar "TRANSITION" 13.1
    bar "STRESS" 16.5
    bar "CRISIS" 33.8
```

### Average Return After Each Regime

```mermaid
bar
    title Average Return (Next 10 Sessions)
    x-axis [Regime, Avg Return %]
    y-axis "Return %" -1 --> 2
    bar "CALM_BULL" 0.48
    bar "CALM_RANGE" 0.21
    bar "CALM_BEAR" 0.25
    bar "TRANSITION" 0.49
    bar "STRESS" 0.72
    bar "CRISIS" 1.63
```

### Worst Return & Downside Risk

```mermaid
xychart-beta
    title "Downside Risk by Regime"
    x-axis [CALM_BULL, CALM_RANGE, CALM_BEAR, TRANSITION, STRESS, CRISIS]
    y-axis "Worst Return %" -30 --> 0
    bar [-6.8, -11.6, -9.1, -17.6, -18.2, -26.5]
    line "Avg Return" [0.48, 0.21, 0.25, 0.49, 0.72, 1.63]
```

### % of Periods Down

```mermaid
pie showData
    title "Probability of Negative 10-Day Return"
    "CALM_BULL (36.7%)" : 36.7
    "CALM_RANGE (39.8%)" : 39.8
    "CALM_BEAR (45.1%)" : 45.1
    "TRANSITION (33.7%)" : 33.7
    "STRESS (33.3%)" : 33.3
    "CRISIS (28.8%)" : 28.8
```

---

## 7. Regime Distribution

```mermaid
pie showData
    title "Regime Frequency (Share of All Days)"
    "CALM_BULL: 27.8%" : 27.8
    "TRANSITION: 25.5%" : 25.5
    "STRESS: 26.0%" : 26.0
    "CALM_RANGE: 10.6%" : 10.6
    "CRISIS: 6.2%" : 6.2
    "CALM_BEAR: 4.1%" : 4.1
```

---

## 8. Overall Assessment Flowchart

```mermaid
flowchart TD
    A[Regime Classifier Assessment] --> B{Volatility Prediction}
    B --> C[Beats coin flip ✓]
    B --> D[Loses to VIX ✗]
    B --> E[Loses to Realized Vol ✗]
    A --> F{Directional Prediction}
    F --> G[No skill ✗]
    F --> H[Worse than always guessing UP ✗]
    A --> I{Risk Detection}
    I --> J[Catches 81% of big drops ✓]
    I --> K[High false positive rate 58% ✗]
    A --> L{Signal Quality}
    L --> M[VIX dominates ✓]
    L --> N[Extra signals add noise ✗]
    L --> O[Entropy ratio weakest ✗]
    
    C --> P[Use case: Educational/Research]
    D --> P
    E --> P
    G --> P
    H --> P
    J --> P
    K --> P
    M --> P
    N --> P
    O --> P
```

---

## 9. Key Takeaways Summary

```mermaid
mindmap
  root((Regime Classifier))
    Volatility Prediction
      Beats coin flip
      Loses to VIX baseline
      Loses to Realized Vol baseline
      STRESS/CRISIS precision 70%
    Turbulent Days
      VIX still wins
      Trade-off: Recall vs Precision
      Classifier flags too many days
    Signal Quality
      VIX: AUC 0.828/0.850 ✓
      Realized Vol: AUC 0.753/0.789 ✓
      Risk Score: No better than RV
      Entropy: Weakest signal
    Big Drops
      Catches 81% of -5% drops
      But 58% false positive rate
    Direction
      NO skill
      Always guess UP wins
    Regime Outcomes
      CRISIS: Highest vol 33.8%
      CRISIS: Best avg return +1.63%
      CRISIS: Lowest % down 28.8%
```

---

## How to View

These diagrams render automatically on GitHub. For local viewing:
- VS Code with Mermaid extension
- Obsidian
- Any Markdown viewer with Mermaid support

---

*Data source: [ACCURACY.md](ACCURACY.md) • Period: Feb 2009 – Oct 2026 (4,457 days) • Educational tool, not investment advice*