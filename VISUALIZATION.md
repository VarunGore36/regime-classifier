# Regime Classifier - Visual Analysis

This document uses Mermaid diagrams (rendered natively on GitHub) to visualize the accuracy data from [ACCURACY.md](ACCURACY.md).

> **Note:** GitHub's Mermaid supports: flowcharts, pie charts, mindmaps, timelines, Gantt, sequence, class, state, ER, user journey, git graphs, quadrant charts, and requirement diagrams. It does **not** support `bar`, `xychart-beta`, or `pie showData`.

---

## 1. Predicting "More Volatile Than Usual" (Next 10 Days)

### Accuracy Comparison

```mermaid
flowchart LR
    subgraph "Accuracy: Predicting High Volatility (Next 10 Days)"
        direction TB
        A1[VIX > Median: 74.4% ✓]
        A2[Realized Vol > Median: 67.7%]
        A3[Classifier STRESS/CRISIS: 63.0%]
        A4[Classifier TRANS/STRESS/CRISIS: 60.4%]
        A5[Coin Flip: 50.0%]
    end
```

### Precision, Recall & Days Flagged

```mermaid
flowchart LR
    subgraph "Classifier: TRANSITION/STRESS/CRISIS"
        C1[Precision: 59.0%] --> C2[Recall: 68.0%] --> C3[Days Flagged: 57.6%]
    end
    subgraph "Classifier: STRESS/CRISIS only"
        D1[Precision: 70.1%] --> D2[Recall: 45.1%] --> D3[Days Flagged: 32.2%]
    end
    subgraph "Baseline: VIX > Median"
        E1[Precision: 74.4%] --> E2[Recall: 74.4%] --> E3[Days Flagged: 50.0%]
    end
    subgraph "Baseline: Realized Vol > Median"
        F1[Precision: 67.7%] --> F2[Recall: 67.7%] --> F3[Days Flagged: 50.0%]
    end
```

### Key Insight Flowchart

```mermaid
flowchart TD
    A[Goal: Predict if next 10 days<br/>more volatile than usual] --> B{Which method?}
    B --> C[Classifier TRANS/STRESS/CRISIS]
    B --> D[Classifier STRESS/CRISIS only]
    B --> E[VIX > Median]
    B --> F[Realized Vol > Median]
    C --> G[Acc: 60.4%<br/>Prec: 59.0%<br/>Rec: 68.0%]
    D --> H[Acc: 63.0%<br/>Prec: 70.1%<br/>Rec: 45.1%]
    E --> I[Acc: 74.4% ✓<br/>Prec: 74.4%<br/>Rec: 74.4%]
    F --> J[Acc: 67.7%<br/>Prec: 67.7%<br/>Rec: 67.7%]
    G --> K[Beats coin flip ✓<br/>Loses to VIX ✗]
    H --> K
    I --> K
    J --> K
```

---

## 2. Predicting "Turbulent" (Top 25% Volatility)

### Accuracy Comparison

```mermaid
flowchart LR
    subgraph "Accuracy: Predicting Turbulent Days (Top 25% Volatility)"
        direction TB
        B1[Always Guess No: 75.0%]
        B2[VIX > Median: 68.3%]
        B3[Classifier STRESS/CRISIS: 71.8%]
        B4[Realized Vol > Median: 64.9%]
        B5[Classifier TRANS/STRESS/CRISIS: 56.5%]
    end
```

### Precision & Recall Comparison

```mermaid
flowchart LR
    subgraph "Precision / Recall Trade-off"
        direction TB
        P1[VIX Median: Prec 43.3% | Rec 86.6%]
        P2[Realized Vol: Prec 39.9% | Rec 79.7%]
        P3[STRESS/CRISIS: Prec 44.9% | Rec 57.8%]
        P4[TRANS/STRESS/CRISIS: Prec 34.0% | Rec 78.3%]
    end
```

### Trade-off Visualization

```mermaid
flowchart LR
    subgraph "High Recall / Low Precision"
        R1[TRANS/STRESS/CRISIS<br/>Recall: 78.3%<br/>Precision: 34.0%<br/>Flags: 57.6%]
        R2[VIX > Median<br/>Recall: 86.6%<br/>Precision: 43.3%<br/>Flags: 50%]
    end
    subgraph "Balanced"
        B1[STRESS/CRISIS only<br/>Recall: 57.8%<br/>Precision: 44.9%<br/>Flags: 32.2%]
        B2[Realized Vol Median<br/>Recall: 79.7%<br/>Precision: 39.9%<br/>Flags: 50%]
    end
```

---

## 3. Signal Ranking Power (AUC)

### AUC Comparison

```mermaid
flowchart LR
    subgraph "AUC: Ranking Power for Volatility Targets"
        direction TB
        S1[VIX Alone: 0.828 / 0.850 ✓]
        S2[Realized Vol 20d: 0.753 / 0.789 ✓]
        S3[Risk Score: 0.749 / 0.783 ~]
        S4[Entropy Ratio ρ: 0.620 / 0.639 ✗]
    end
```

### Detailed AUC Breakdown

```mermaid
flowchart LR
    subgraph "High-Vol Target AUC"
        H1[VIX: 0.828]
        H2[Realized Vol: 0.753]
        H3[Risk Score: 0.749]
        H4[Entropy: 0.620]
    end
    subgraph "Turbulent Target AUC"
        T1[VIX: 0.850]
        T2[Realized Vol: 0.789]
        T3[Risk Score: 0.783]
        T4[Entropy: 0.639]
    end
    subgraph "Rank Correlation"
        R1[VIX: 0.67]
        R2[Realized Vol: 0.52]
        R3[Risk Score: 0.50]
    end
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

### Detection Breakdown

```mermaid
pie
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
flowchart LR
    subgraph "Directional Prediction Accuracy"
        direction TB
        D1[Market always UP (baseline): 64.8%]
        D2[CALM_BULL predicts UP: 63.3%]
        D3[PCA Structure Score: 52.8%]
        D4[Coin Flip: 50.0%]
        D5[CALM_BEAR predicts DOWN: 45.1%]
    end
```

### Signal Performance Matrix

```mermaid
flowchart LR
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
flowchart LR
    subgraph "Average Realized Volatility (Next 10 Sessions)"
        direction TB
        V1[CRISIS: 33.8%]
        V2[STRESS: 16.5%]
        V3[TRANSITION: 13.1%]
        V4[CALM_BEAR: 12.6%]
        V5[CALM_RANGE: 12.0%]
        V6[CALM_BULL: 11.4%]
    end
```

### Average Return After Each Regime

```mermaid
flowchart LR
    subgraph "Average Return (Next 10 Sessions)"
        direction TB
        R1[CRISIS: +1.63%]
        R2[STRESS: +0.72%]
        R3[TRANSITION: +0.49%]
        R4[CALM_BULL: +0.48%]
        R5[CALM_BEAR: +0.25%]
        R6[CALM_RANGE: +0.21%]
    end
```

### Worst Return & Downside Risk

```mermaid
flowchart LR
    subgraph "Downside Risk by Regime"
        direction TB
        W1[CRISIS: -26.5% | Avg: +1.63%]
        W2[STRESS: -18.2% | Avg: +0.72%]
        W3[TRANSITION: -17.6% | Avg: +0.49%]
        W4[CALM_RANGE: -11.6% | Avg: +0.21%]
        W5[CALM_BEAR: -9.1% | Avg: +0.25%]
        W6[CALM_BULL: -6.8% | Avg: +0.48%]
    end
```

### % of Periods Down

```mermaid
pie
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
pie
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