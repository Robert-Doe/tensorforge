# Module 11 — DECISIONS.md

## Decision 1: Wrap SVM in a Pipeline with StandardScaler
**Decision:** Always pair SVC with StandardScaler inside a sklearn Pipeline.
**Why:** SVM optimises a margin measured in Euclidean distance. Features on different scales (evidence_score 0-10, days_open 0-365) produce wildly distorted distances. Scaling makes every feature contribute equally.
**Trade-off:** Pipeline adds a layer of indirection; beginners must understand that model.fit() now fits both the scaler and the SVM.

## Decision 2: Enable probability=True
**Decision:** Pass `probability=True` to SVC.
**Why:** SVM's native output is a class label, not a probability. Setting probability=True uses Platt scaling (a calibrated sigmoid on the margin distance) to produce probability estimates — needed for ROC-AUC comparison with other models.
**Trade-off:** Slightly slower training (~2x) due to additional cross-validation step inside Platt scaling.

## Decision 3: Demonstrate C-sweep instead of grid search
**Decision:** Manually sweep C values and print a table, rather than using GridSearchCV.
**Why:** Seeing the accuracy change at each C value builds intuition for the bias-variance trade-off. C=small → wide margin → underfitting; C=large → narrow margin → overfitting. GridSearchCV hides this curve.
**Trade-off:** Manual sweep only covers one hyperparameter; GridSearchCV would tune C and gamma jointly.

## Decision 4: Include both linear and RBF kernels
**Decision:** Show both `kernel="linear"` and `kernel="rbf"` side by side.
**Why:** Linear SVM is interpretable (has a weight vector); RBF SVM handles non-linear boundaries at the cost of interpretability. Comparing them on the same dataset illustrates when the kernel trick adds value.
**Trade-off:** Two models to explain; adds cognitive load, but the conceptual payoff is worth it at this stage.
