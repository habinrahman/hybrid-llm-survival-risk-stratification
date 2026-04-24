# Hybrid LLM–Survival Model for Early Patient Risk Stratification

---

## 1. Cover Page

| Field | Details |
|--------|---------|
| **Project Title** | Hybrid LLM–Survival Model for Early Patient Risk Stratification |
| **Author** | Shaima Rauf (NNM24CSE18) |
| **Degree Program** | Master of Technology (M.Tech), Computer Science and Engineering |
| **Institution** | NMAM Institute of Technology, Nitte |
| **Department** | Department of Computer Science and Engineering |
| **Guide** | Dr. Vijay Murari |
| **Date** | April 2026 |

---

## 2. Executive Summary

This project implements a **multimodal survival analysis system** for **early patient risk stratification**. It combines **structured electronic health record (EHR)** variables (demographics, vitals, laboratory values) with **unstructured clinical text** (physician-style notes) by encoding text through a **biomedical transformer** (BioClinicalBERT), **fusing** modalities via **dimensionality reduction and concatenation**, and training **Cox proportional hazards** and **DeepSurv** (deep neural partial-likelihood) models to predict **time-to-event** outcomes under **right censoring**.

**Objectives and significance:** Hospital workflows generate both tabular signals and narrative documentation. Classical survival models often ignore text; generic NLP pipelines may ignore temporal censoring. This work bridges **survival analysis** and **clinical language models** in a reproducible pipeline suitable for research and prototyping.

**Key contributions:** (1) **Leakage-aware** preprocessing and fusion (fit on training only). (2) **DeepSurv survival functions** via **Breslow-type baseline** cumulative hazard for calibration-style metrics. (3) **Optuna** hyperparameter search script. (4) **Console** and **Streamlit** inference interfaces. (5) **Modular** package layout under `src/hybrid_survival/`.

**Real-world applications:** Mortality or readmission risk screening, prioritization for follow-up, and analytics pipelines—subject to regulatory validation and use of real de-identified data.

---

## 3. Problem Statement

**Challenges in early risk stratification:** Clinicians must integrate heterogeneous information quickly; delayed identification of high-risk patients increases adverse outcomes and resource strain.

**Limitations of traditional survival models:** Cox models assume **multiplicative hazards** and often **linear** effects on the log-hazard scale; they may underfit complex interactions unless heavily engineered. They historically rely on **structured covariates only**, missing semantic cues in notes.

**Role of clinical text:** Notes capture symptoms, course, and clinician reasoning not always coded in structured fields. **Large language models (LLMs)** and **clinical encoders** provide dense representations of text but must be combined with survival-appropriate training and **evaluation under censoring**.

---

## 4. Project Objectives

| Type | Goals |
|------|--------|
| **Primary** | Build an end-to-end pipeline: structured preprocessing + clinical embeddings + fusion + Cox/DeepSurv training + discrimination/calibration metrics. |
| **Primary** | Prevent **data leakage** by fitting preprocessors and fusion on **training** data only. |
| **Secondary** | Provide **survival probabilities** from DeepSurv (not only risk scores). |
| **Secondary** | Enable **hyperparameter tuning** (Optuna), **console** and **web** prediction, and **artifact persistence**. |
| **Research** | Demonstrate a **multimodal survival** baseline comparable in spirit to recent medical deep survival and transformer-survival literature. |

**Expected outcomes:** Saved models, CSV metric tables, calibration and survival curve plots, and reproducible configuration via YAML.

---

## 5. Key Features of the Project

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Multimodal data integration** | Joint use of tabular EHR features and free-text clinical notes per patient. |
| 2 | **Structured feature processing** | Median/mode imputation, standardization of numerics, label encoding of categoricals (`StructuredDataPreprocessor`). |
| 3 | **BioClinicalBERT embeddings** | HuggingFace `emilyalsentzer/Bio_ClinicalBERT` encoder; **[CLS]** pooling by default. |
| 4 | **Feature fusion** | Optional **PCA** on text embeddings, per-modality **standardization**, **horizontal concatenation** with structured features. |
| 5 | **Cox proportional hazards** | `lifelines.CoxPHFitter` with elastic-net-style penalization parameters from config. |
| 6 | **DeepSurv neural network** | MLP predicting log-risk; **negative log partial likelihood** training with mini-batches. |
| 7 | **Survival probability prediction** | Cox: lifelines survival functions. DeepSurv: **Breslow** baseline cumulative hazard on training set, then \(S(t|x)=\exp(-H_0(t)\exp(\eta))\). |
| 8 | **C-index and Brier score** | Harrell’s **concordance index** for discrimination; **Brier** at configured time horizons when \(S(t\mid x)\) is available. |
| 9 | **Calibration and survival curves** | Plots saved under `results/` (non-interactive Matplotlib backend). |
| 10 | **Console prediction** | `scripts/predict_console.py` for interactive single-patient inference. |
| 11 | **Streamlit web application** | `app/streamlit_app.py` for form-based input and charts. |
| 12 | **Hyperparameter optimization** | `scripts/tune_hyperparameters.py` (Optuna) writes `configs/config.best.yaml`. |
| 13 | **Model persistence** | Pickle/joblib artifacts for models, preprocessors, fusion, optional `text_pca.pkl`, embedder settings. |
| 14 | **Reproducibility** | Global seed helper for Python, NumPy, and PyTorch; fixed random splits in config. |
| 15 | **Modular architecture** | Package `hybrid_survival` with submodules: `data`, `features`, `models`, `evaluation`, `pipelines`, `inference`, `utils`. |

---

## 6. System Architecture

The system ingests a cohort dataframe, **splits** into train/validation/test **before** any supervised feature fitting, learns **tabular preprocessing** and **fusion** on training data only, encodes notes with a **frozen** pretrained encoder (weights not adapted to labels—avoiding label leakage from val/test), trains **Cox** and **DeepSurv**, and evaluates with **risk-based C-index** and **probability-based** metrics where survival functions exist.

### ASCII diagram (conceptual)

```text
Structured Data ─────┐
                     ├──► Feature Fusion ───► Survival Models ───► Predictions
Clinical Notes ──────┘              │                    │
   (BioClinicalBERT)                 │                    ├── Cox PH
                                     ▼                    └── DeepSurv
                              Evaluation Metrics
                              (C-index, Brier, plots)
```

**Data flow (high level):**

1. **Load cohort** → synthetic or future CSV/MIMIC-derived tables.  
2. **Stratified split** → train / validation / test indices.  
3. **Structured `fit`** on train → `transform` on val/test.  
4. **Text encoding** → embeddings per split (same pretrained weights).  
5. **Fusion `fit`** on train structured+text → `transform` on val/test.  
6. **Train** Cox and DeepSurv on fused train; optional **early stopping** on val **C-index** for DeepSurv.  
7. **Evaluate** on fused test; **save** artifacts.

---

## 7. Project Directory Structure

> **Note:** The codebase was refactored into a **package-first** layout. Legacy monolithic filenames (`data_preprocessing.py`, etc.) now correspond to **modules inside** `src/hybrid_survival/`. The table below maps **documentation names** to **current paths**.

### 7.1 Repository tree (current)

```text
llm/
├── configs/
│   └── config.yaml              # Main YAML configuration
├── src/
│   └── hybrid_survival/
│       ├── __init__.py
│       ├── inference.py       # Load artifacts + single-patient prediction
│       ├── data/
│       │   └── preprocessing.py   # ← data_preprocessing.py (conceptual)
│       ├── features/
│       │   ├── text_embeddings.py # ← text_embeddings.py
│       │   └── fusion.py          # ← feature_fusion.py
│       ├── models/
│       │   └── survival.py        # ← survival_models.py
│       ├── evaluation/
│       │   └── metrics.py         # ← model_evaluation.py
│       ├── pipelines/
│       │   └── hybrid_pipeline.py # ← main orchestration (main_pipeline logic)
│       └── utils/
│           └── repro.py
├── scripts/
│   ├── train.py
│   ├── predict_console.py
│   └── tune_hyperparameters.py
├── app/
│   └── streamlit_app.py
├── notebooks/
│   └── demo_notebook.ipynb    # Jupyter demo
├── models/                      # Created at runtime (gitignored typical)
├── results/
├── data/
├── run_pipeline.py
├── main_pipeline.py
├── requirements.txt
├── README.md
├── PROJECT_DOCUMENTATION.md     # This file
├── GETTING_STARTED.md
└── PROJECT_STRUCTURE.md
```

### 7.2 File-by-file reference

| Documented name | Current location | Role |
|-----------------|------------------|------|
| **config.yaml** | `configs/config.yaml` | Model IDs, training split, fusion, survival hyperparameters, evaluation horizons, paths. |
| **data_preprocessing.py** | `src/hybrid_survival/data/preprocessing.py` | `MIMICDataLoader`, `StructuredDataPreprocessor`. |
| **text_embeddings.py** | `src/hybrid_survival/features/text_embeddings.py` | `ClinicalBERTEmbedder`, `extract_text_features`. |
| **feature_fusion.py** | `src/hybrid_survival/features/fusion.py` | `FeatureFusion`, `MultimodalDataset`. |
| **survival_models.py** | `src/hybrid_survival/models/survival.py` | `CoxModel`, `DeepSurvModel`, Breslow baseline for DeepSurv. |
| **model_evaluation.py** | `src/hybrid_survival/evaluation/metrics.py` | `SurvivalMetrics`, `ModelEvaluator`, `print_model_comparison_table`, CV helper. |
| **main_pipeline.py** | Root `main_pipeline.py` + `src/.../hybrid_pipeline.py` | Entry point delegates to `HybridSurvivalPipeline`. |
| **run_pipeline.py** | Root `run_pipeline.py` | CLI-friendly full train/eval/save. |
| **requirements.txt** | Root | Python dependencies. |
| **demo_notebook.ipynb** | `notebooks/demo_notebook.ipynb` | Minimal demo; adds `src` to path and runs pipeline. |
| **README.md** | Root | User-oriented overview and commands. |
| **scripts/predict_console.py** | `scripts/predict_console.py` | Interactive inference. |
| **app/streamlit_app.py** | `app/streamlit_app.py` | Streamlit UI. |
| **scripts/train.py** | `scripts/train.py` | argparse wrapper around pipeline. |
| **scripts/tune_hyperparameters.py** | `scripts/tune_hyperparameters.py` | Optuna study + `config.best.yaml`. |

---

## 8. Technology Stack

| Category | Technology |
|----------|------------|
| Programming language | Python (3.10+ recommended) |
| Deep learning framework | PyTorch |
| Survival analysis | Lifelines (Cox PH, concordance) |
| Clinical language encoder | BioClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) |
| NLP library | Hugging Face Transformers |
| Data processing | Pandas, NumPy |
| Machine learning utilities | scikit-learn (imputation, scaling, PCA, splits) |
| Visualization | Matplotlib, Seaborn (where used in evaluation module) |
| Web framework | Streamlit |
| Configuration | YAML (PyYAML) |
| Serialization | pickle, joblib |
| Hyperparameter search | Optuna |
| Notebooks | Jupyter, ipykernel |

---

## 9. Dataset Description

**Current default:** The loader builds a **synthetic, MIMIC-style** cohort for demonstration: demographics, vitals, labs, templated **clinical notes**, a **length-of-stay**-like time variable, and a binary **event** (e.g., mortality or 30-day readmission depending on configuration).

**Features (illustrative):** `age`, `gender`, `ethnicity`, `insurance`, `admission_type`, vital signs, common labs, `clinical_notes`, `time`, `event`.

**Targets:**  
- **`time`:** time to event or censoring (here aligned with synthetic LOS).  
- **`event`:** 1 if event observed, 0 if censored.

**Future work:** Replace `MIMICDataLoader.load_cohort` with pipelines joining **MIMIC-III / MIMIC-IV** tables (admissions, patients, chartevents, noteevents) under institutional data agreements and **de-identification** policies.

---

## 10. Methodology

### Step 1 — Data acquisition and split  
Load dataframe → **stratified** train/validation/test split **before** supervised preprocessing.

### Step 2 — Structured preprocessing  
**Fit** on training only: median imputation for numerics, most-frequent for categoricals, **standardization** for numerics, **label encoding** for categoricals. **Transform** validation and test.

### Step 3 — Clinical text embedding  
Tokenize notes (max length from config, typically 512). Forward pass through **BioClinicalBERT**; extract **[CLS]** vector per note (768 dimensions for the default checkpoint). Optional light **text preprocessing** (e.g., lowercasing) before encoding.

### Step 4 — Multimodal fusion  
**Fit** on training: optional **PCA** on text embeddings to `fusion.text_dim_reduction` dimensions; **standardize** structured and (reduced) text blocks; **concatenate** into a single feature vector per patient. **Transform** val/test with the same fitted objects.

### Step 5 — Survival modeling  
- **Cox:** partial likelihood with penalization.  
- **DeepSurv:** stochastic gradient optimization of partial likelihood–style loss; **early stopping** can track validation **C-index**; **Breslow** baseline estimated on training risk scores and event structure.

### Step 6 — Model evaluation  
**C-index** from risk scores; **Brier scores** at `evaluation.time_horizons` using predicted survival probabilities; **calibration** and **survival curve** plots when \(S(t\mid x)\) is implemented.

### Step 7 — Deployment interfaces  
**Console** script and **Streamlit** app load the same artifacts (`struct_preprocessor`, `fusion`, models) and repeat inference-time transforms.

---

## 11. Machine Learning Models

### A. Cox proportional hazards model

**Formulation:** For individual \(i\), hazard at time \(t\):

\[
h_i(t) = h_0(t) \exp(\mathbf{x}_i^\top \boldsymbol{\beta})
\]

where \(h_0(t)\) is an **unspecified baseline hazard**, \(\mathbf{x}_i\) are covariates (here fused features), and \(\boldsymbol{\beta}\) are estimated by **partial likelihood** maximization (with optional **penalization** in implementation).

**Advantages:** Interpretable coefficients, well-understood inference, natively provides **survival curves** via lifelines.

**Limitations:** Proportional hazards assumption may be violated; limited modeling of nonlinear interactions without explicit feature engineering.

### B. DeepSurv neural network

**Architecture:** Feedforward network with **linear–BatchNorm–ReLU–Dropout** blocks per hidden layer, ending in a **scalar** output interpreted as **log-relative-risk** \(\eta_i\).

**Loss function:** Training minimizes a **negative log partial likelihood**–style objective computed on **mini-batches** (sorted by time within batch). This is a **stochastic** approximation to full Cox partial likelihood.

**Training methodology:** Adam optimizer; configurable learning rate, batch size, epochs; **GPU** used when available; **early stopping** on validation **concordance** when enabled in `configs/config.yaml`.

**Survival prediction:** After training, **Breslow-type** increments estimate baseline cumulative hazard \(H_0(t)\) on the training sample; then

\[
S(t \mid \mathbf{x}_i) = \exp\big(-H_0(t)\,\exp(\eta_i)\big).
\]

---

## 12. LLM Text Embedding Module

**BioClinicalBERT** is a **BERT-base**–style encoder continued pretraining on **MIMIC-III** clinical text, distributed as `emilyalsentzer/Bio_ClinicalBERT` on HuggingFace.

**Process:**  
1. String preprocessing (optional).  
2. **WordPiece tokenization** with special tokens; padding/truncation to `max_sequence_length`.  
3. Transformer layers produce contextual hidden states.  
4. **Pooling:** default **\[CLS\]** token embedding as the document vector.

**Embedding dimension:** **768** (hidden size of base BERT architecture).

**Optimization opportunities (research):** mean pooling over tokens, chunk-long documents, domain adapters, or fine-tuning with survival-aware objectives (beyond current frozen encoder).

---

## 13. Evaluation Metrics

### Concordance index (C-index)

**Idea:** Among comparable pairs \((i,j)\) where the lower-risk patient has longer observed time or the event ordering is consistent with risk, the fraction correctly ordered.

**Interpretation:** 0.5 is random; 1.0 is perfect discrimination. Implementation uses **lifelines** `concordance_index` with appropriate sign convention so that **higher predicted risk** aligns with shorter survival.

### Brier score (at fixed time \(t^\ast\))

Using predicted survival \(S(t^\ast \mid x_i)\) and observed status at \(t^\ast\), a **proper** score would use inverse probability of censoring weighting (**IPCW**). The project includes a **simplified** Brier implementation suitable for demos; for publication-grade calibration, consider IPCW (e.g., via dedicated survival libraries).

**Interpretation:** Lower is better (better calibration sharpness at \(t^\ast\)).

### Calibration curve

Bins predicted survival at \(t^\ast\) vs observed survival proxy; closeness to the **45° line** indicates good calibration.

### Survival curves

\(S(t)\) vs \(t\) for individual patients or subsets—useful for communicating **dynamic risk** over time.

---

## 14. Terminal Output (sample)

The pipeline prints a **Model Comparison** block via `print_model_comparison_table`. Numeric values **depend on your run**; the layout matches the following template:

```text
------------------------------------------------------------
Model Comparison
------------------------------------------------------------
Model     C-index   N Samples   N Events   Event Rate
Cox       0.8710    1000        300           30.00%
DeepSurv  0.8920    1000        300           30.00%

Brier Scores:
t=2.4: Cox=0.0811 | DeepSurv=0.0790
t=4.8: Cox=0.1708 | DeepSurv=0.1682
t=9.7: Cox=0.2282 | DeepSurv=0.2255
------------------------------------------------------------
```

> **Note:** Exact C-index and Brier values vary with **data**, **seed**, and **hardware**. The example numbers illustrate **presentation formatting**, not guaranteed performance on the synthetic cohort.

---

## 15. Console-Based Prediction

**Script:** `scripts/predict_console.py`

**Workflow:** After training, the script loads `configs/config.yaml` and artifacts from `models/` (`struct_preprocessor`, `fusion`, `cox_model`, `deepsurv_model`). The user enters **numeric** vitals/labs (blank allowed for missing), **categorical** fields, and a **clinical note**. The primary block prints **DeepSurv risk** and **survival probabilities** at each configured horizon; a **Cox reference** line may be printed.

**Example interaction (illustrative):**

```text
Enter patient fields (leave numeric blank for missing).
age: 72
heart_rate: 88
...
clinical_notes: Patient admitted with shortness of breath; started on oxygen; monitoring closely.
------------------------------------------------------------
Patient Risk Prediction
------------------------------------------------------------
Risk Score: 0.8421
Survival Probability (t=2.4): 0.9123
Survival Probability (t=4.8): 0.8012
Survival Probability (t=9.7): 0.6534
------------------------------------------------------------
(Cox reference risk score: 0.8102)
```

---

## 16. Streamlit Web Application

**File:** `app/streamlit_app.py`

**Features:**  
- Sidebar-style inputs for structured fields and a **text area** for notes.  
- **Predict** button runs the same fusion + model stack as the console tool.  
- Displays **risk metric** and **survival probabilities** at YAML horizons.  
- Plots a **DeepSurv survival curve** over a dense time grid for visualization.  
- Optional expander for **Cox** reference metrics.

**Run:**

```bash
streamlit run app/streamlit_app.py
```

**Prerequisite:** Train once so `models/` contains the expected pickle/joblib files.

---

## 17. Installation and Setup

```powershell
# From project root (e.g., llm/)
python -m venv venv

# Activate (Windows PowerShell)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**First run:** HuggingFace will **download** BioClinicalBERT weights (network required).

---

## 18. Execution Commands

| Action | Command |
|--------|---------|
| Train + evaluate + save | `python run_pipeline.py` |
| Same (alternate entry) | `python main_pipeline.py` |
| Train via script module | `python scripts/train.py` |
| Hyperparameter tuning | `python scripts/tune_hyperparameters.py --trials 25` |
| Console prediction | `python scripts/predict_console.py` |
| Streamlit UI | `streamlit run app/streamlit_app.py` |
| Jupyter notebook | `jupyter notebook notebooks/demo_notebook.ipynb` |

> **Import path:** Entry scripts add `src/` to `sys.path` so that `import hybrid_survival` resolves correctly.

---

## 19. Expected Outputs

| Output | Location / description |
|--------|-------------------------|
| **Saved models** | `models/cox_model.pkl`, `models/deepsurv_model.pkl` |
| **Preprocessing & fusion** | `models/struct_preprocessor.pkl`, `models/fusion_module.pkl`, optional `models/text_pca.pkl` |
| **Embedder metadata** | `models/embedder_config.pkl` |
| **Metrics CSV** | `results/model_comparison.csv` |
| **Figures** | `results/*_calibration.png`, `results/*_survival_curves.png`, `results/deepsurv_training.png` |
| **Tuning** | `configs/config.best.yaml` (after Optuna) |

---

## 20. Results and Discussion

**Interpretation:** Higher **C-index** indicates better **ranking** of patients by risk. **Brier** at clinically meaningful horizons summarizes **error** of survival probability predictions at those times (subject to the simplified censoring handling noted above).

**Multimodal benefit:** Text embeddings can capture narrative severity and context **orthogonal** to structured labs; fusion allows Cox/DeepSurv to **jointly** weight tabular and semantic signals. Empirical gains must be validated on **real** data with **proper baselines** (structured-only, text-only, and full multimodal).

---

## 21. Limitations

| Limitation | Impact |
|------------|--------|
| **Synthetic cohort** | Metrics are **demonstrations**, not clinical claims. |
| **Simplified Brier** | Not full **IPCW**; interpret cautiously for research write-ups. |
| **Compute** | ClinicalBERT encoding and DeepSurv training benefit from **GPU**; CPU runs are slower. |
| **Regulatory** | Not a **medical device**; no deployment validation (FDA/CE) in this repository. |
| **Proportional hazards** | Cox and DeepSurv PH-style survival may misfit if hazards cross. |

---

## 22. Future Enhancements

- Integrate **real MIMIC-III/IV** extraction and time-aligned notes.  
- **Transformer survival heads** (e.g., discrete-time hazards, attention over tokens).  
- **Federated learning** across sites with privacy constraints.  
- **Explainability:** SHAP for tabular branch, attention or saliency for text (where applicable).  
- **Cloud deployment** (Docker, FastAPI + autoscaling) with audit logging.  
- **IPCW Brier** and time-dependent **AUC** for rigorous reporting.

---

## 23. Real-World Applications

- **Hospital readmission** and **mortality** risk triage.  
- **ICU** length-of-stay or deterioration alerts (with appropriate monitoring data).  
- **Population health** analytics and cohort screening (governance required).

---

## 24. References

1. Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B*.  
2. Katzman, J. L., et al. (2018). **DeepSurv:** personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Medical Research Methodology*.  
3. Alsentzer, E., et al. (2019). Publicly available clinical BERT embeddings. *NAACL Clinical NLP Workshop*.  
4. Harrell, F. E., et al. (1982). Evaluating the yield of medical tests. *JAMA* — concordance / C-index foundations.  
5. Graf, E., et al. (1999). Assessment and comparison of prognostic classification schemes for survival data. *Statistics in Medicine* — prediction error concepts.  
6. Johnson, A. E. W., et al. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*.  
7. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* (context for clinical time series, if extended).  
8. Wolf, T., et al. (2020). HuggingFace’s Transformers: State-of-the-art natural language processing. *EMNLP System Demonstrations*.

---

## 25. Reproducibility and Model Persistence

| Artifact | Purpose |
|----------|---------|
| `cox_model.pkl` | Trained Cox PH fitter wrapper + feature names. |
| `deepsurv_model.pkl` | PyTorch weights, architecture hyperparameters, training history, Breslow baseline arrays. |
| `fusion_module.pkl` | Fitted PCA (if any), scalers, fusion method configuration. |
| `struct_preprocessor.pkl` | Imputers, scaler, label encoders fit on **training** tabular data only. |
| `text_pca.pkl` | Optional duplicate of PCA object for convenience (when used). |
| `embedder_config.pkl` | Model name and tokenization limits for inference consistency. |

**Reproducibility levers:** `training.random_state` in YAML; `set_global_seed`; fixed Optuna study seed (can be extended in script).

---

## 26. Conclusion

This project delivers a **presentation-ready**, **modular** multimodal survival pipeline: **structured EHR processing**, **BioClinicalBERT** note embeddings, **PCA-concatenation fusion**, **Cox** and **DeepSurv** models with **DeepSurv survival functions**, **leakage-aware fitting**, **Optuna** tuning support, and **console/Streamlit** interfaces. It is well suited as an **M.Tech** artifact demonstrating integration of **clinical NLP** and **survival analysis**, with a clear path to **stronger empirical claims** via real MIMIC data and IPCW metrics.

---

## 27. Project Readiness Score

**Score: 8 / 10**

| Strength | Rationale |
|----------|-----------|
| Architecture & engineering | Clear package layout, config-driven runs, inference paths, saved artifacts. |
| Methods | Standard survival + deep Cox-style network + clinical LM embeddings. |
| Gaps for 9–10 | Real-data ingestion, unit/integration tests, IPCW Brier, regulatory/clinical validation narrative. |

---

## Appendix A. Quick Presentation Notes (Viva / Slides)

1. **One-liner:** Multimodal survival model combining **tabular EHR** and **BioClinicalBERT** text for **time-to-event** prediction.  
2. **Why it matters:** Notes carry information not in structured codes; survival models respect **censoring**.  
3. **Innovation:** **No leakage** pipeline (fit preprocessors/fusion on **train** only); **DeepSurv** **survival curves** via **Breslow** baseline.  
4. **Models:** **Cox** for interpretability; **DeepSurv** for nonlinear risk.  
5. **Metrics:** **C-index** for discrimination; **Brier** at fixed horizons; **calibration plots**.  
6. **Demo:** Synthetic cohort today; **MIMIC** tomorrow.  
7. **Live demo:** Run `run_pipeline.py`, show `results/`, then **Streamlit** or **console** prediction.  
8. **Honest limits:** Synthetic data, simplified Brier, not a medical device.  
9. **Future:** Real MIMIC joins, transformer survival, IPCW, explainability, cloud deploy.  
10. **Closing:** Strong **software + methods** foundation for a research-oriented M.Tech project.

---

*End of document.*
