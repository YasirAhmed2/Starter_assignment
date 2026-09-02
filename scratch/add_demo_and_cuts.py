import json
from pathlib import Path

demo_outline_source = [
    "## 13. 5-Minute Showcase Demo Outline (Week-8 Ready)\n",
    "\n",
    "*A concise 5-minute presentation outline structured for the Week-8 showcase.*\n",
    "\n",
    "---\n",
    "\n",
    "### Minute 1: The Question & Problem Framing\n",
    "- **The FlyRank Content Challenge:** FlyRank builds content as infrastructure across tens of thousands of published URLs. Over time, content traffic decays as search algorithms evolve and user intent shifts.\n",
    "- **The Decision Bottleneck:** Editorial capacity is strictly finite (teams can only review 200–500 articles per month). Relying on hand-crafted rules (e.g., age > 180 days) yields only 38.0% Precision@100 — worse than the 54.2% portfolio decay base rate.\n",
    "- **The Core Question:** *Out of 30,000 managed URLs, which specific decaying pages should human editors fix FIRST to maximize traffic recovery ROI?*\n",
    "\n",
    "### Minute 2: Methodology & Leakage-Free Validation\n",
    "- **Data Scope:** 79 million search impression events aggregated into 30,000 unique URLs across 32 client domains with 90-day baseline performance windows.\n",
    "- **Leakage Defense:** Excluded 11 post-observation and label-derived columns (`trend_direction`, `trend_pct`, product output flags).\n",
    "- **Validation Design:** 5-Fold GroupKFold strictly grouped by client domain (`client_id`) — guaranteeing zero cross-client information leakage into validation folds.\n",
    "\n",
    "### Minute 3: One Key Chart (Precision@K Comparison)\n",
    "- **Chart:** Precision@K across models vs. Rule Baseline (`charts/capstone_pak_comparison.svg`).\n",
    "- **Headline Visual:** At operational queue depth (Top 100 articles), HistGradientBoosting achieves **87.0% Precision@100** vs. **38.0%** for the Rule Baseline — delivering a **+49.0 percentage point lift**.\n",
    "- **Takeaway:** Editors acting on the ML queue spend 87% of their time on truly decaying pages, compared to only 38% on the heuristic queue.\n",
    "\n",
    "### Minute 4: One Honest Result & Operational Boundaries\n",
    "- **Model Performance:** HistGradientBoosting achieved an out-of-fold ROC-AUC of **0.6944** and PR-AUC of **0.6964**.\n",
    "- **Honest Nuance:** This is an observational ranking model, not a causal guarantee of traffic recovery. Google core algorithm shifts and competitor moves introduce unmodeled variance.\n",
    "- **Cold-Start Boundary:** 1,328 URLs (4.43%) with zero 90-day baseline impressions cannot be scored by performance signals and are routed to a separate discovery pipeline.\n",
    "\n",
    "### Minute 5: One Practical Recommendation (Action Playbook)\n",
    "- **Risk-Calibrated Triage:** Rather than dumping raw probabilities onto editors, classify URLs into four action tiers (`refresh`, `refresh_and_review_ctr`, `refresh_and_review_engagement`, `monitor`) and four review urgency levels.\n",
    "- **Operational Impact:** Automates safe metadata/CTR updates on low-risk pages (saving 44% manual review overhead) while routing high-stakes revenue pages to human copywriters and SMEs.\n",
    "- **Strict No-Go Policy:** Zero unedited automated LLM overwrites; zero unsupervised page deletions or redirects.\n"
]

shareable_cuts_source = [
    "## 14. Shareable Cuts of the Work\n",
    "\n",
    "*Public-safe, honest summaries ready for social sharing and employer portfolio reviews.*\n",
    "\n",
    "---\n",
    "\n",
    "### Cut 1: Methodology Social Post (LinkedIn / X)\n",
    "\n",
    "```text\n",
    "How do you prioritize content refreshes across 30,000 URLs when editorial capacity is capped at 300 articles a month?\n",
    "\n",
    "In my latest ML capstone with FlyRank, I built and validated a machine learning ranking system on 79M production search records to predict organic traffic decay.\n",
    "\n",
    "Key takeaway: Hand-crafted heuristic rules (e.g. age > 180 days) achieved only 38.0% Precision@100 (lagging the 54.2% portfolio decay base rate). By training a gradient-boosted classifier with 5-fold domain-grouped cross-validation to prevent client leakage, we achieved 87.0% Precision@100 (+49% lift) and an out-of-fold ROC-AUC of 0.6944.\n",
    "\n",
    "We translated these predictions into an operational triage playbook that automates safe metadata updates on low-risk pages while routing high-stakes refreshes to human editors.\n",
    "\n",
    "Full paper & reproducible notebooks: https://yasirahmed2.github.io/Starter_assignment/\n",
    "#MachineLearning #DataScience #SearchEngineering #MLOps #FlyRank\n",
    "```\n",
    "\n",
    "---\n",
    "\n",
    "### Cut 2: 3-Sentence Employer-Facing Summary\n",
    "\n",
    "> **What I built:** An end-to-end machine learning ranking and editorial triage pipeline that predicts organic search traffic decay and prioritizes content refresh sprints.  \n",
    "> **On what data:** An anonymized production dataset of 79 million search impression records aggregated across 30,000 URLs and 32 client domains, validated with 5-fold domain-grouped cross-validation to eliminate cross-client leakage.  \n",
    "> **What it showed:** A gradient-boosted classifier achieved 87.0% Precision@100 (+49.0 percentage point lift over rule-based heuristics) and an ROC-AUC of 0.6944, enabling an operational workflow that reduces manual editorial review overhead by 44% while protecting top-tier search revenue.\n"
]

target_files = [
    "work/notebooks/capstone.ipynb",
    "work/notebooks/w07_action_playbook.ipynb"
]

for nb_path in target_files:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    # Remove previous demo outline or cuts if present
    nb["cells"] = [c for c in nb["cells"] if not any(x in "".join(c.get("source", [])) for x in ["5-Minute Showcase Demo Outline", "Shareable Cuts of the Work"])]
    
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": demo_outline_source
    })
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": shareable_cuts_source
    })
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully updated {nb_path}")
