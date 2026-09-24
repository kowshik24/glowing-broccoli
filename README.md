# glowing-broccoli — machine failure prediction

Predicting `Machine failure` on the AI4I 2020 predictive maintenance dataset, then checking
with SHAP that the model is actually picking up on the right physics for each failure mode
instead of just fitting noise. Started as an EDA notebook, grew into a full pipeline once I
started digging into why the minority class metrics were so bad.

Dataset: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset (also
sitting in `dataset/ai4i2020.csv`, 10k rows, synthetic but modeled on real industrial sensor data).

## Why this is harder than it looks

10,000 rows, only 339 are actual failures (3.39%). And `Machine failure` isn't one thing — it's
the OR of five separate, mostly-independent failure mechanisms:

- **TWF** (tool wear failure) — tool crosses a wear-time window, 120 cases
- **HDF** (heat dissipation failure) — air/process temp gap too small + low rpm, 115 cases
- **PWF** (power failure) — torque × rpm outside a power window, 95 cases
- **OSF** (overstrain failure) — tool wear × torque past a threshold, 98 cases
- **RNF** (random failure) — literally a coin flip, 19 cases, no physical cause at all

The model only ever sees the combined `Machine failure` flag, never which mode caused it. RNF
is the interesting one — since it has zero real signal, if SHAP shows the model actually leaning
on some feature for RNF cases, that's a red flag that it's fitting noise. If it shows no clear
signal, that's actually the model getting it right.

## What's in the notebook

`notebooks/Exploratory-Data-Analysis.ipynb`, roughly in this order:

1. EDA — distributions, class balance, correlation heatmap, failure-mode counts, scatter plots
   that visually confirm the physics rules from the dataset docs (torque vs rpm for PWF, temp
   gap vs rpm for HDF, etc.)
2. Feature engineering — `Power`, `Temp_diff`, `Overstrain`, `Overstrain_ratio`, built directly
   from the failure-mode definitions rather than generic polynomial features
3. Feature extraction — drops `UDI`/`Product ID` (identifiers), and **drops TWF/HDF/PWF/OSF/RNF
   from the training features entirely** since they're literally components of the target —
   training on them would be leakage. They're kept aside only for the XAI cross-check later.
4. Stratified 70/20/10 train/val/test split
5. Preprocessing (scale + one-hot) fit on train only, then SMOTE-family oversampling on train only
6. A quick bake-off between SMOTE / BorderlineSMOTE / ADASYN on validation F1 — BorderlineSMOTE
   won this run and gets used for the rest of the notebook (rerun it and it might pick something
   else depending on the random draw, that's fine, the notebook adapts automatically)
7. PCA, mostly to show *why* linear models are going to struggle — the two classes overlap almost
   completely in a 2D projection, so the failure boundary isn't linearly separable
8. Six classical models (Logistic Regression, SVM, KNN, Decision Tree, Random Forest, XGBoost)
   plus a small PyTorch MLP, all evaluated on the untouched imbalanced val/test splits
9. Test set evaluation with ROC/PR curves and confusion matrices for everything
10. Hyperparameter search (`RandomizedSearchCV`, F1-scored) + threshold tuning on the winning
    model (XGBoost), because the default 0.5 cutoff is a bad choice when 96% of your data is one
    class
11. 5-fold CV to sanity-check the tuned model isn't just lucky on one 1,000-row test split
12. A calibration check — `predict_proba` on a tree model isn't automatically trustworthy as an
    actual probability, worth confirming before leaning on it for the threshold above
13. SHAP explanations on the final tuned model, cross-checked feature-by-feature against
    TWF/HDF/PWF/OSF/RNF to see if the model actually learned the right mechanism per failure type
14. A bonus multi-label model that predicts the five failure modes directly instead of inferring
    them from SHAP — more useful in practice (tells you *what to fix*, not just *that something's
    wrong*)
15. Saves the tuned model + preprocessor to `models/` for the inference script below

Results move around a bit run to run (random search, SMOTE variant, etc.), but the last full run
landed the tuned XGBoost at precision 1.00 / recall 0.82 / F1 0.90 on the test set's 34 failure
cases, up from precision 0.86 / recall 0.88 / F1 0.87 before tuning. All five SHAP-vs-failure-mode
checks passed — tool wear drives TWF, temp gap/rpm drives HDF, power drives PWF, overstrain drives
OSF, and RNF correctly shows no dominant feature.

## Setup

Using [uv](https://github.com/astral-sh/uv) for the venv, but plain `venv` + `pip install -r
requirements.txt` works too if you don't have it.

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt

# register the kernel so Jupyter/VS Code can find it
python -m ipykernel install --user --name glowing-broccoli-venv --display-name "Python 3 (venv)"
```

Then open `notebooks/Exploratory-Data-Analysis.ipynb` and pick the `Python 3 (venv)` kernel, or
run it headless:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 \
  --ExecutePreprocessor.kernel_name=glowing-broccoli-venv \
  Exploratory-Data-Analysis.ipynb
```

Give it 15-20 minutes on a laptop — the hyperparameter search alone is ~200 XGBoost fits, plus
the 5-fold CV re-fits a fresh model per fold.

### A gotcha if you're on Apple Silicon

numpy, scikit-learn, xgboost and torch each bundle their own threaded BLAS/OpenMP runtime, and
running all of them in one process (which a Jupyter kernel does) reliably crashed the kernel here
with no useful error — nbconvert just reports "Kernel died" with no traceback pointing at the real
cause. Took a while to track down. The imports cell forces everything single-threaded
(`OMP_NUM_THREADS=1` etc, plus `KMP_DUPLICATE_LIB_OK=TRUE`) before anything else gets imported —
if you strip that out expect random segfaults partway through, usually right around the first
`torch.tensor()` call after xgboost has already been used. Also: skip PyTorch's MPS backend for
this — the model's tiny, CPU is plenty fast, and MPS + xgboost together in one process wasn't
stable either.

If xgboost itself won't import on macOS (`Library not loaded: @rpath/libomp.dylib`), you need
`brew install libomp` — it's not something pip can fix.

## Using the trained model without the notebook

Once the notebook's run at least once (it saves `models/preprocessor.joblib`,
`models/xgboost_tuned.joblib`, `models/metadata.joblib`):

```bash
python src/predict.py --csv path/to/new_readings.csv
python src/predict.py --csv path/to/new_readings.csv --out predictions.csv
```

Input CSV needs: `Type`, `Air temperature [K]`, `Process temperature [K]`,
`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`. The script reconstructs the same
engineered features (`Power`, `Temp_diff`, `Overstrain`, `Overstrain_ratio`) the model was trained
on and applies the tuned decision threshold, not the default 0.5.

## Layout

```
dataset/ai4i2020.csv          the raw data
notebooks/                    the actual work — EDA through XAI
models/                       saved model + preprocessor (generated by the notebook)
src/predict.py                CLI inference on new data
requirements.txt              pinned versions, tested on macOS/arm64
```
