# Capstone Report — Lane 2: Refresh Opportunity Scoring

- **Author:** Yasir Ahmed
- **Lane:** Lane 2 — Refresh Opportunity Scoring
- **Repo:** [https://github.com/YasirAhmed2/Starter_assignment](https://github.com/YasirAhmed2/Starter_assignment)
- **Deployed Paper:** [https://yasirahmed2.github.io/Starter_assignment/](https://yasirahmed2.github.io/Starter_assignment/)
- **Date:** March 2026

---

## 1. Problem framing

FlyRank operates content as infrastructure: autonomously researching, generating, publishing, and monitoring articles at scale across client websites. Over time, content naturally experiences organic search traffic decay due to evolving search intent, decaying click-through rates (CTR), and search engine algorithm updates.

However, enterprise editorial bandwidth is strictly finite — content teams can only refresh ~200 to 500 articles per month out of portfolio libraries containing tens of thousands of URLs. The core operational decision is: **out of thousands of managed URLs, which specific decaying pages should human editors fix FIRST to maximize organic search traffic recovery?**

- **Unit of Analysis:** A unique pseudonymized URL (`content_id`).
- **Output:** Calibrated probability of traffic decay (`prob_decay`), a composite priority score (0–100), an action classification (`refresh`, `refresh_and_review_ctr`, `refresh_and_review_engagement`, `monitor`), and a human review urgency tier.
- **Action Taken:** Editorial sprint routing — automating low-risk metadata/CTR refreshes while routing high-stakes, revenue-driving decaying assets to human copywriters and subject-matter experts (SMEs).
- **Cost of Wrong Calls:**
  - *False Positives (over-flagging healthy pages):* Wastes scarce copywriter and SME time editing pages that are already performing well.
  - *False Negatives (ignoring decaying pages):* Permits compounding organic traffic loss, eroding client search authority and revenue.
- **Why ML Helps:** Existing production heuristics rely on hand-crafted rules (e.g. `days_since_last_update > 180` AND `ctr < 0.02`), which achieve only 38.0% Precision@100 (lagging behind the 54.21% portfolio decay base rate). A learned gradient-boosted model captures non-linear interactions across freshness, impression volume, position drift, and engagement to deliver 87.0% Precision@100 (+49.0 percentage point lift).

---

## 2. Data safety & leakage audit

- **Data Source:** Anonymized FlyRank production search dataset comprising ~79 million search impression records aggregated into 30,000 unique URLs across 32 client domains over 90-day baseline performance windows.
- **Leakage Prevention & Excluded Columns:** 11 post-observation and label-derived columns were strictly excluded from feature inputs:
  - `trend_direction`, `trend_pct` (target-defining variables)
  - `impressions_last_30d`, `clicks_last_30d` (future observation window signals)
  - `health_score`, `quick_win_flag`, `needs_attention_flag` (pre-existing product output flags, preventing circular logic)
  - `content_id`, `client_id` (identifiers reserved for grouping/stratification only)
- **Public Safety Pass:** All client names are anonymized as cryptographic IDs (`client_001` ... `client_050`). All raw search queries, destination URLs, and brand tokens are completely omitted.

---

## 3. Baseline

- **Baseline Design:** A transparent, composite domain-heuristic score simulating FlyRank's hand-crafted product rule:
  $$\text{baseline\_score} = 0.4 \times \text{stale\_score} + 0.3 \times \text{low\_ctr\_score} + 0.3 \times \text{position\_loss\_score}$$
- **Baseline Performance (5-Fold GroupKFold OOF):**
  - **P@10:** 60.0%
  - **P@20:** 45.0%
  - **P@50:** 44.0%
  - **P@100:** 38.0%
  - **P@500:** 43.6%
- **Baseline Limitation:** Hand-crafted thresholds miss multi-dimensional decay signals and over-index on raw article age, yielding a Precision@100 (38.0%) that performs worse than the random base rate (54.21%).

---

## 4. Model & feature engineering

- **Target Definition:** Binary indicator `is_declining_label` ($=1$ if organic search traffic direction is `down` over the subsequent 90-day evaluation window, 0 otherwise). Portfolio base rate is **54.21%** (16,264 / 30,000 URLs).
- **Candidate Models:**
  1. Logistic Regression (L2 regularized baseline)
  2. Decision Tree (max_depth=5)
  3. Random Forest (100 estimators)
  4. HistGradientBoostingClassifier (Champion: max_iter=150, lr=0.05, max_leaf_nodes=31)
- **Feature Set (10 Non-Leaking Predictors):**
  - *Freshness:* `days_since_last_update`, `word_count`
  - *Traffic & Search Volume:* `impressions_90d`, `clicks_90d`
  - *Engagement & CTR:* `ctr`, `engagement_rate`, `scroll_rate`
  - *SERP Ranking:* `avg_position`, `position_tier`
  - *Content Categorization:* `content_type` (One-Hot Encoded)

---

## 5. Evaluation & validation audit

- **Validation Split:** 5-Fold GroupKFold grouped strictly by `client_id` (32 domains). This guarantees that no client domain appears simultaneously in train and test splits, reflecting real-world generalization to new customer sites.
- **Model Comparison Table (5-Fold GroupKFold OOF):**

| Model | ROC-AUC | PR-AUC | P@10 | P@20 | P@50 | P@100 | P@500 |
|---|---|---|---|---|---|---|---|
| Rule Baseline | N/A | N/A | 60.0% | 45.0% | 44.0% | 38.0% | 43.6% |
| Logistic Regression | 0.5730 | 0.5939 | 40.0% | 55.0% | 54.0% | 59.0% | 62.8% |
| Decision Tree (d=5) | 0.6580 | 0.6510 | 10.0% | 10.0% | 22.0% | 29.0% | 57.2% |
| Random Forest | 0.6619 | 0.6557 | 80.0% | 60.0% | 60.0% | 54.0% | 64.4% |
| **HistGradientBoosting (Champion)** | **0.6944** | **0.6964** | **90.0%** | **95.0%** | **92.0%** | **87.0%** | **77.8%** |

- **Error Analysis:**
  - False Positives at lower thresholds typically occur on high-authority evergreen URLs that withstand aging without traffic loss.
  - False Negatives occur on seasonal topics or sudden algorithm shifts where 90-day historical features showed healthy baseline engagement prior to rapid SERP displacement.

---

## 6. Interpretation & signal audit

- **Permutation Importance Top Drivers:**
  1. `days_since_last_update` (Mean importance drop: +0.082 AUC)
  2. `impressions_90d` (+0.054 AUC)
  3. `ctr` (+0.041 AUC)
  4. `avg_position` (+0.033 AUC)
  5. `engagement_rate` (+0.021 AUC)
- **Surprise / Negative Finding:** Raw `word_count` exhibited near-zero marginal predictive power once impression volume and position tier were controlled for. Article length alone does not protect against content decay.

---

## 7. Recommendations & operational playbook

1. **Portfolio Triage Policy:**
   - **Tier 1 — Low-Risk Auto-Refresh (44% of Queue):** Automate title tag and snippet metadata updates for URLs with high decay confidence but low traffic exposure.
   - **Tier 2 — Mandatory Human Review (56% of Queue):** High-traffic decaying assets and core pillar content are routed directly to copywriters/SMEs with structured diagnostic reason codes (`decay_prob`, `traffic_loss_risk`).
2. **Strict Prohibited Automations (The No-Go List):**
   - NO automated generative LLM re-writes directly pushed to production without human review.
   - NO automatic 301 redirects, content merges, or page deletions.
   - NO programmatic title swaps on primary converting landing pages.
3. **Monitoring & Retraining Triggers:**
   - Trigger retraining if domain decay base rate shifts by $>10\%$, if mean prediction drift exceeds 0.08, or after major Google core algorithm updates.

---

## 8. Reproducibility & data credit

- **Reproduction Steps:**
```bash
git clone https://github.com/YasirAhmed2/Starter_assignment.git
cd Starter_assignment
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace work/notebooks/capstone.ipynb
```
- **Environment:** Python 3.11+, scikit-learn 1.3+, pandas 1.5+, numpy, matplotlib. Seed = 42.
- **Data Credit:** Built on the **[FlyRank ML Internship dataset](https://flyrank.ai/)**.

---

## 9. 5-Minute Showcase Demo Outline (Week-8 Ready)

- **Minute 1 — Question:** FlyRank manages content as infrastructure across tens of thousands of URLs. When content decays, how do we prioritize which pages human editors should fix first when capacity is capped at 300 refreshes/month?
- **Minute 2 — Method:** 79M search records across 30,000 URLs and 32 domains. Strict 5-Fold GroupKFold by domain with zero cross-client leakage and all post-observation signals removed.
- **Minute 3 — Chart:** Figure 1 (`charts/capstone_pak_comparison.svg`) — HistGradientBoosting achieves 87.0% Precision@100 vs. 38.0% for the baseline rule (+49.0% absolute lift).
- **Minute 4 — Honest Result:** Out-of-fold ROC-AUC of 0.6944 and PR-AUC of 0.6964. Observational ranking without causal overclaims; 1,328 cold-start items (4.4%) routed to discovery queue.
- **Minute 5 — Recommendation:** Operational triage cuts review overhead by 44% via automated metadata updates on low-risk pages while focusing human copywriters on high-stakes revenue drivers.

---

## 10. Shareable Cuts of the Work

### Cut 1: Short Social Post (Methodology & Public-Safe)
```text
How do you prioritize content refreshes across 30,000 URLs when editorial capacity is capped at 300 articles a month?

In my latest ML capstone with FlyRank, I built and validated a machine learning ranking system on 79M production search records to predict organic traffic decay.

Key takeaway: Hand-crafted heuristic rules (e.g. age > 180 days) achieved only 38.0% Precision@100 (lagging the 54.2% portfolio decay base rate). By training a gradient-boosted classifier with 5-fold domain-grouped cross-validation to prevent client leakage, we achieved 87.0% Precision@100 (+49% lift) and an out-of-fold ROC-AUC of 0.6944.

We translated these predictions into an operational triage playbook that automates safe metadata updates on low-risk pages while routing high-stakes refreshes to human editors.

Full paper & reproducible notebooks: https://yasirahmed2.github.io/Starter_assignment/
#MachineLearning #DataScience #SearchEngineering #MLOps #FlyRank
```

### Cut 2: 3-Sentence Employer-Facing Summary
> **What I built:** An end-to-end machine learning ranking and editorial triage pipeline that predicts organic search traffic decay and prioritizes content refresh sprints.  
> **On what data:** An anonymized production dataset of 79 million search impression records aggregated across 30,000 URLs and 32 client domains, validated with 5-fold domain-grouped cross-validation to eliminate cross-client leakage.  
> **What it showed:** A gradient-boosted classifier achieved 87.0% Precision@100 (+49.0 percentage point lift over rule-based heuristics) and an ROC-AUC of 0.6944, enabling an operational workflow that reduces manual editorial review overhead by 44% while protecting top-tier search revenue.
