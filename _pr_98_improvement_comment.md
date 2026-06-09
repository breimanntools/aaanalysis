# PR #98 Improvement: Flexible Model Selection for CV Gate

## Issue
The current implementation uses a hardcoded `RandomForestClassifier` for the cross-validation gate, which can be slow for large feature sets and many candidates. SVM and Logistic Regression are often faster alternatives for this use case.

## Suggested Improvement

**Add a new parameter `model`** that allows users to choose the classifier for the CV gate:

```python
def simplify(self,
             df_feat: pd.DataFrame = None,
             labels: ut.ArrayLike1D = None,
             X: Optional[ut.ArrayLike2D] = None,
             strategy: str = "greedy",
             model: Union[str, BaseEstimator] = "rf",  # NEW
             cv: int = 5,  # NEW (or rename n_cv to cv for consistency)
             # ... other parameters
             ):
```

### Details:

1. **String options** (predefined fast models):
   - `"rf"` (default): `RandomForestClassifier(random_state=random_state)`
   - `"svm"`: `SVC(random_state=random_state, probability=True)` with balanced class weights
   - `"log_reg"`: `LogisticRegression(random_state=random_state, max_iter=1000)`
   - Default to `"rf"` for backward compatibility, but `"log_reg"` is recommended for speed

2. **Direct model objects** (sklearn-compatible):
   - Users can pass their own fitted or unfitted estimator
   - Must implement `fit(X, y)` and `predict(X)` / `score(X, y)`
   - Example: `model=SVC(kernel='poly', C=0.1)`

3. **Implementation**:
   - Add validation in `check_model_cv()` to ensure the model or string is valid
   - Update `_score_feature_set_()` to accept the model as a parameter
   - Update backend call in `simplify_cpp_()` to pass the model through

### Example Usage:
```python
# Fast logistic regression (default in future)
df_simplified = cpp.simplify(df_feat=df_feat, labels=labels, model="log_reg", cv=5)

# Custom SVM with specific hyperparameters
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
df_simplified = cpp.simplify(df_feat=df_feat, labels=labels, model=model, cv=5)

# Keep RF (current default)
df_simplified = cpp.simplify(df_feat=df_feat, labels=labels, model="rf", cv=5)
```

### Benefits:
- ✅ Users can optimize for speed vs. predictive power
- ✅ Backward compatible (RF is default)
- ✅ Flexible enough for any sklearn classifier
- ✅ Consistent with scikit-learn API conventions

### Files to Update:
1. `aaanalysis/feature_engineering/_cpp.py` — add `model` parameter to `simplify()` docstring and validation
2. `aaanalysis/feature_engineering/_backend/cpp/_simplify.py` — update `_score_feature_set_()` and `simplify_cpp_()` to use flexible model
3. `aaanalysis/utils.py` — add `LIST_CV_MODELS = ["rf", "svm", "log_reg"]` constant
4. `tests/unit/cpp_tests/test_cpp_simplify.py` — add tests for different model selections

---

**This follows sklearn conventions and gives users full control over the speed/accuracy tradeoff!**
