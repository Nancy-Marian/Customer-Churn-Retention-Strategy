# Customer Churn Prediction and Retention Strategy (Online Retail II)

> From 1M+ messy transactions to a retention decision: data cleaning → churn model validated on a later time period → cost-benefit targeting with sensitivity analysis.
> Author: [TODO: your name] · [TODO: LinkedIn / GitHub link]

## TL;DR

- Predicts which customers will not buy in the next 90 days. Tested on a **later time period** than it was trained on: ROC-AUC **0.79** (all customers) and **0.76** for active customers, whose riskiest 10% churn about **2.0x** as often as average.
- A plain logistic regression performs as well as XGBoost (confidence intervals overlap), so the simpler model is a valid choice.
- Recency and the gap between orders drive churn. About a quarter of customers are gone within 30 days of their first order.
- **A blanket 10% discount would lose about £166K per quarter** on this customer base. Discount-free outreach is profitable under every assumption tested, and discounts pay only for a small, high-risk group.

## Business problem

A UK online gift wholesaler wants to keep customers who are about to stop buying. Which customers should the retention team contact, with what kind of offer, and is it worth the cost?

## Data

UCI *Online Retail II*: 1,067,371 invoice lines, Dec 2009 to Dec 2011. The raw file is genuinely messy: 22.8% of rows have no Customer ID, 34,335 exact duplicates, 19,494 cancelled-invoice rows, 22,950 rows with zero or negative quantity, 6,207 with zero or negative price, and non-product rows (postage, fees, vouchers, test entries).

| Cleaning step | Rows before | Rows after | Removed |
|---|---|---|---|
| Raw | 1,067,371 | 1,067,371 | 0 |
| Drop exact duplicates | 1,067,371 | 1,033,036 | 34,335 |
| Drop non-product stock codes | 1,033,036 | 1,027,215 | 5,821 |
| Move cancellations to a returns table | 1,027,215 | 1,009,301 | 17,914 |
| Drop quantity ≤ 0 or price ≤ 0 | 1,009,301 | 1,003,338 | 5,963 |
| Drop missing Customer ID (guest checkouts) | 1,003,338 | 776,577 | 226,761 |

Result: 776,577 sale lines, 5,852 customers, 36,594 invoices, 4,619 products, revenue about £17.07M. Revenue peaks every October to November (pre-Christmas).

## Approach

| Notebook | What it does |
|---|---|
| `01_data_prep.ipynb` | Audit, logged cleaning, EDA, churn label and 20+ behavioural features at two snapshots |
| `02_model.ipynb` | Baseline vs logistic regression vs XGBoost, tested on the future with bootstrap confidence intervals, calibration check, SHAP |
| `03_insights_and_business.ipynb` | Survival analysis, cohort retention, statistical tests, revenue at risk, targeting list, sensitivity analysis |
| `churn_utils.py` | Shared functions (feature builder, confidence intervals, probability shift) |

Design choices:
- **Churn definition:** no purchase in the 90 days after a cutoff date. Features use only data before the cutoff (an assertion checks there is no leakage).
- **Out-of-time validation:** train on the 2011-06-10 snapshot, test on 2011-09-09. The training label window ends before the test cutoff.
- **Imbalance:** class weights rather than SMOTE.
- **"Active" customers** are those who bought in the last 180 days. They are the ones a retention team can still reach, so results are reported for both groups.

## Results

| Test group | Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Top-decile lift |
|---|---|---|---|---|
| All customers (5,253; churn 56.6%) | Baseline | 0.500 | 0.566 | 0.95 |
| | Logistic regression | 0.792 (0.780 to 0.805) | 0.812 (0.797 to 0.827) | 1.59 |
| | XGBoost | 0.791 (0.778 to 0.802) | 0.806 (0.790 to 0.821) | 1.59 |
| Active customers (2,775; churn 38.5%) | Baseline | 0.500 | 0.385 | 0.90 |
| | Logistic regression | 0.747 (0.729 to 0.765) | 0.623 (0.595 to 0.660) | 1.92 |
| | XGBoost | 0.759 (0.742 to 0.780) | 0.645 (0.614 to 0.676) | 1.98 |

- On active customers XGBoost looks slightly better, but the intervals overlap, so I would not claim a winner.
- **Calibration:** the churn rate fell from 68.0% (June snapshot) to 56.6% (September), so the raw model over-predicted (60.6% average). Shifting probabilities to the expected churn rate lowered the Brier score from 0.1853 to 0.1841. The model stays somewhat overconfident at the extremes.
- **Drivers** (permutation importance and SHAP agree): recency (importance 0.064), then average gap between orders (0.025), then total items, spend, and frequency. Return count and return rate push predictions toward churn once purchase volume is accounted for.

![SHAP summary](images/shap_summary.png)
![Calibration](images/calibration.png)

## Findings

1. **The first order is the biggest leak.** The survival curve falls about 24% within 30 days of a first order. Median customer lifetime is 427 days, with 72% still active at 90 days and 54% at 365. A larger first order lowers the risk of leaving (hazard ratio 0.81, p < 0.001).
2. **One-time buyers churn at 79.6% vs 46.8% for repeat buyers** (difference +32.8 points, 95% CI +30.2 to +35.3, Cramér's V = 0.30). Not adjusted for recency or tenure.
3. **Cohort retention is low:** about 21% of new customers buy again in month 1 and 18% in month 6, with autumn seasonality visible.
4. **Customers who returned items churn less** (41.7% vs 67.0%, V = 0.25). This reflects that returners are heavy buyers, not an effect of returns.
5. **47% of customers have been silent for more than 180 days.** They hold about 46% of the revenue at risk, which is a win-back opportunity separate from retention.
6. Recency differs strongly between churned and retained customers (median 284 vs 65 days; a random churner has the longer recency 76% of the time).

![Survival curves](images/survival_curve.png)
![Cohort retention](images/cohort_heatmap.png)

## Business recommendation

Expected 90-day revenue for the September 2011 customer base is £2.60M, of which about **£699K (27%) is at risk**.

Benefit of contacting a customer = P(churn) × save rate × 90-day value × margin − discount paid to everyone who buys − contact cost. Assumptions: 30% margin, £0.50 per contact. Figures are for the 2,775 active customers over 90 days.

| Offer | Save rate | Customers worth contacting | Net benefit |
|---|---|---|---|
| No discount (email or call) | 5% | 2,440 | +£4,285 |
| No discount | 10% | 2,663 | +£9,846 |
| 5% discount | 20% | 772 | +£2,889 |
| 10% discount | 20% | 211 | +£267 |
| **10% discount to all active customers** | any | 2,775 | **-£166,268** |

- **Start with discount-free outreach.** It is profitable at every save rate tested.
- Reserve discounts for the small group with high churn risk and high value.
- Rank customers by expected benefit (risk × value), not by risk alone. Contacting the top 20% by risk lost about £650.
- Focus onboarding on getting the **second order** within 30 days.

![Profit curve](images/profit_curve.png)

## How I would test the campaign

The save rate drives every number above, and it is an assumption. To measure it:
1. Randomly split eligible customers into **treatment** (contacted) and **control** (held out).
2. Primary metric: share who order within 90 days. Guardrail: margin per customer.
3. Example: detecting a rise from 40% to 45% with 80% power and α = 0.05 needs about 1,500 customers per group. That is more than half of the 2,775 active customers, so the test would run in waves or detect only larger effects.
4. Report the result with a confidence interval and update the save rate above.

## Limitations and assumptions

- Margin (30%), save rate, discount, contact cost and the 180-day "lapsed" cutoff are **placeholders**, not facts about this retailer. The sensitivity table shows how conclusions change.
- Value over 90 days is the customer's average historical spend. For one-time buyers this equals their whole spend, which probably overstates it.
- The probability shift uses the test period's actual churn rate (hindsight). In production, use the latest completed window and re-check as seasons change.
- One retailer, two years, strong Q4 seasonality. Business figures are estimates, not realised results.
- Correlations (returns, first-order value) are not causal effects.

## Future work

Hyperparameter tuning and experiment tracking (Optuna, MLflow), a scoring API, drift monitoring, a lifetime-value model, and uplift modelling once campaign test data exists.

## How to run

```bash
pip install -r requirements.txt
# put online_retail_II.xlsx in data/raw/, then run the notebooks in order:
# 01_data_prep → 02_model → 03_insights_and_business
```

## Repository structure

```
data/raw/            original Excel file (not committed)
data/processed/      cleaned data, features, scored customers, targeting list
models/              saved model
images/              charts used in this README
01_data_prep.ipynb · 02_model.ipynb · 03_insights_and_business.ipynb
churn_utils.py · requirements.txt
```
