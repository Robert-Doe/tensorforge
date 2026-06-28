# Module 12 — DECISIONS.md

## Decision 1: RandomForest before XGBoost
**Decision:** Use sklearn's GradientBoostingClassifier rather than xgboost.
**Why:** xgboost requires a separate install and has a different API. sklearn's built-in GB conveys the same concepts (sequential trees, learning rate, residuals) without adding a dependency. XGBoost is introduced conceptually but not coded here.
**Trade-off:** sklearn GB is 5-10x slower than XGBoost on large datasets; acceptable for 120-sample toy data.

## Decision 2: n_jobs=-1 for RandomForest only
**Decision:** Pass n_jobs=-1 to RandomForestClassifier but not GradientBoostingClassifier.
**Why:** RF trees are independent and can be grown in parallel. GB trees are sequential by definition — each tree depends on the previous ensemble's errors. Passing n_jobs to GB would have no effect and would confuse learners about why parallelism works for RF but not GB.
**Trade-off:** None — this is technically correct and pedagogically clarifying.

## Decision 3: Show feature importances for both models
**Decision:** Print and plot feature_importances_ for RF and GB separately.
**Why:** RF and GB assign importance differently (RF averages across all trees equally; GB weights earlier trees' splits more). Comparing them side by side shows that "feature importance" is not a single fixed property of the data but a model-specific measure.
**Trade-off:** Two plots; more output. Worth it for the insight.

## Decision 4: Keep max_depth=3 for GradientBoosting weak learners
**Decision:** Use shallow trees (max_depth=3) as GB weak learners.
**Why:** Boosting works best with weak learners — trees that are individually just slightly better than random. Deep trees in GB overfit quickly because each round already focuses on hard examples. max_depth=3 is the standard default in research and production.
**Trade-off:** May underfit on datasets with complex interaction effects; increase max_depth if training loss stalls.
