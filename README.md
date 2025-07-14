# Time-Series-Forecasting-for-Retail-Demand-Prediction-and-Strategy

This repository presents our end-to-end solution for the [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) competition, focusing on high-resolution **daily sales prediction** for thousands of Walmart products across multiple stores and states.

---

## Objective

To build an accurate, scalable machine learning pipeline that forecasts unit sales for Walmart items across 28 days using historical sales, pricing, calendar, and event data.

---

## Methodology

Adopted a **batch-wise modeling strategy**, engineering extensive lag and rolling features per product-store combination. Our modeling approach combined **XGBoost**, **LightGBM**, and ensemble baselines with validation aligned to the competition’s rolling forecast window strategy.

### Key Steps:

1. **Data Cleaning & Anomaly Checks**
2. **Wide-to-Long Transformation**
3. **Batch-wise Feature Engineering**  
   - Lag features: `lag_7`, `lag_28`
   - Rolling statistics: `rolling_mean_7`, `rolling_std_7`, etc.
   - Temporal: `dayofweek`, `month`, `year`
   - Price features: `sell_price`, `price_change`
4. **Time-based Validation Split** (last 28 days)
5. **Model Training & Comparison**  
   - XGBoost (best), LightGBM, Random Forest, Ridge, KNN, MLP
6. **Evaluation using RMSE, RMSSE, WRMSSE**
7. **Kaggle Submission**

---

## Final Results

| Metric            | Score         |
|------------------|---------------|
| **Best Model** | XGBoost       |
| Validation RMSE  | 2.1818        |
| Mean RMSSE       | 2249.72       |
| WRMSSE (Kaggle)  | **13,180.29** |
| Leaderboard Rank | Top 20%       |

**Top Features:**  
- `sales_roll_mean_7`  
- `sales_roll_mean_14`  
- `sell_price`  
- `lag_28`  
- `dayofweek`

---

## Tech Stack

| Tool/Library        | Purpose                           |
|---------------------|------------------------------------|
| **Python (Pandas, NumPy)** | Data manipulation             |
| **XGBoost & LightGBM** | Core regression models         |
| **Scikit-learn**     | Metrics, preprocessing            |
| **Matplotlib/Seaborn** | Visualization                   |
| **TQDM**             | Progress tracking                 |
| **Google Colab / Jupyter** | Development environment      |

---

## Key Learnings

- Feature engineering had greater performance impact than hyperparameter tuning.
- Batch-wise modeling avoided memory overload while improving specificity.
- Historical trends and price features were dominant predictors of demand.

