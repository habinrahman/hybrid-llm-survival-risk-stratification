# Hybrid LLM-Survival Model for Early Patient Risk Stratification

This project implements a multimodal deep learning approach for patient risk stratification by combining structured Electronic Health Record (EHR) data with unstructured clinical notes using Large Language Models (LLMs) and survival analysis.

## Project Overview

The system integrates:
- **Structured EHR Data**: Demographics, vital signs, lab values
- **Clinical Text**: Physician notes processed via ClinicalBERT embeddings
- **Survival Models**: Cox Proportional Hazards and DeepSurv neural networks

## Architecture

```
EHR Data
├── Structured Features → Preprocessing → Feature Matrix
│                                             ↓
└── Clinical Notes → ClinicalBERT → Embeddings → Feature Fusion → Survival Models
                                                                    ├── Cox PHM
                                                                    └── DeepSurv
```

## Features

### Data Processing
- **Structured Data Preprocessing**: Imputation, normalization, encoding
- **Text Embedding Generation**: ClinicalBERT-based semantic representations
- **Multimodal Fusion**: Combines structured and text features

### Survival Models
- **Cox Proportional Hazards**: Classical statistical survival model
- **DeepSurv**: Deep learning-based survival analysis with neural networks

### Evaluation
- **Concordance Index (C-index)**: Discrimination metric
- **Brier Score**: Calibration metric
- **Calibration Curves**: Visual assessment of model calibration
- **Cross-Validation**: Robust performance estimation

## Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (optional, for faster training)

### Layout

```
configs/config.yaml       # Hyperparameters and paths
src/hybrid_survival/      # Library code (data, features, models, evaluation, pipelines)
scripts/                  # train, tune, console prediction
app/streamlit_app.py      # Web UI
notebooks/                # Jupyter demos
```

### Setup

```bash
cd llm   # project root
python -m venv venv
# Windows: venv\Scripts\activate
# Unix:   source venv/bin/activate
pip install -r requirements.txt
```

### Download ClinicalBERT Model

The pipeline automatically downloads the ClinicalBERT model from HuggingFace on first run. Ensure you have internet connection for the initial setup.

## Quick Start

### 1. Train and evaluate (recommended)

```bash
python run_pipeline.py
# or
python scripts/train.py
```

### 2. Python API

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

from hybrid_survival.pipelines.hybrid_pipeline import HybridSurvivalPipeline

pipeline = HybridSurvivalPipeline(config_path="configs/config.yaml")
results = pipeline.run_full_pipeline()
```

### 3. Step-by-step (no leakage)

```python
pipeline.load_data(outcome="mortality")
pipeline.split_raw_data()
pipeline.preprocess_structured_data()
pipeline.generate_text_embeddings()
pipeline.fuse_features()
pipeline.train_models()
pipeline.evaluate_models()
pipeline.visualize_results()
pipeline.save_models()
```

### 4. Hyperparameter tuning (Optuna)

```bash
python scripts/tune_hyperparameters.py --trials 25
```

Writes `configs/config.best.yaml` — point `--config` at it for `scripts/train.py`.

### 5. Console inference

After training:

```bash
python scripts/predict_console.py
```

### 6. Streamlit

```bash
streamlit run app/streamlit_app.py
```

### 7. Advanced reporting (after training)

Training with `python run_pipeline.py` also produces (when `advanced.enabled` is true in `configs/config.yaml`):

| Output | Description |
|--------|-------------|
| `logs/project.log` | Timestamped pipeline log |
| `results/model_comparison.png` | Bar chart of C-index (Cox vs DeepSurv) |
| `results/kaplan_meier_curve.png` | Empirical KM curve on the **test** cohort |
| `results/shap_summary_deepsurv.png` | SHAP summary for DeepSurv (subset for speed) |
| `results/prediction_report.pdf` | Example PDF for the first test patient |

Console: risk tier, normalized category, and interval summaries are printed after evaluation.

Set `advanced.run_shap: false` to skip SHAP (faster runs).

### 8. FastAPI REST service

From project root (after training so `models/` exists):

```bash
pip install fastapi uvicorn
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

- `GET /` — health message  
- `POST /predict` — JSON body matching `PatientData` (see `api.py`): structured fields + `clinical_notes`; returns `risk_score`, `risk_category`, survival probabilities, and Cox risk.

### 3. Using Custom Data

To use your own MIMIC-III data:

1. Update the `data_path` in `config.yaml`
2. Ensure data follows the expected format (see Data Format section)
3. Run the pipeline

## Configuration

Edit `configs/config.yaml` (paths, `fusion.text_dim_reduction`, `training.*`, `survival.*`, `evaluation.time_horizons`).

## Data Format

### MIMIC-III Structure

The pipeline expects data with the following columns:

**Required Columns:**
- `subject_id`: Patient identifier
- `hadm_id`: Hospital admission identifier
- `clinical_notes`: Free-text clinical notes
- `event`: Binary event indicator (1=event, 0=censored)
- `time`: Time to event or censoring

**Structured Features:**
- Demographics: `age`, `gender`, `ethnicity`, `insurance`
- Vitals: `heart_rate`, `systolic_bp`, `diastolic_bp`, `temperature`, `respiratory_rate`, `spo2`
- Labs: `glucose`, `creatinine`, `hemoglobin`, `wbc`

### Example Data

```python
import pandas as pd

df = pd.DataFrame({
    'subject_id': [1, 2, 3],
    'age': [65, 45, 70],
    'gender': ['M', 'F', 'M'],
    'clinical_notes': [
        'Patient admitted with chest pain...',
        'Post-operative recovery...',
        'Admitted for CHF exacerbation...'
    ],
    'event': [1, 0, 1],
    'time': [5.2, 10.5, 3.1]
})
```

## Module Documentation

Add `src` to `PYTHONPATH` or `sys.path`, then import from `hybrid_survival`.

### 1. Data (`hybrid_survival.data.preprocessing`)

```python
from hybrid_survival.data.preprocessing import StructuredDataPreprocessor, MIMICDataLoader

loader = MIMICDataLoader("./data")
df = loader.load_cohort(outcome="mortality")
preprocessor = StructuredDataPreprocessor()
X_struct = preprocessor.fit_transform(df, numeric_features, categorical_features)
```

### 2. Text (`hybrid_survival.features.text_embeddings`)

```python
from hybrid_survival.features.text_embeddings import ClinicalBERTEmbedder

embedder = ClinicalBERTEmbedder(model_name="emilyalsentzer/Bio_ClinicalBERT")
embeddings = embedder.encode_corpus(clinical_notes)
```

### 3. Fusion (`hybrid_survival.features.fusion`)

```python
from hybrid_survival.features.fusion import FeatureFusion, MultimodalDataset

fusion = FeatureFusion(fusion_method="concatenation", text_dim_reduction=64)
X_fused = fusion.fit_transform(X_struct, X_text)
dataset = MultimodalDataset(X_struct, X_text, y_event, y_time)
```

### 4. Survival (`hybrid_survival.models.survival`)

```python
from hybrid_survival.models.survival import CoxModel, DeepSurvModel

cox = CoxModel(penalizer=0.01)
cox.fit(X_train, y_event_train, y_time_train)

deepsurv = DeepSurvModel(input_dim=X_train.shape[1], hidden_layers=[128, 64, 32])
deepsurv.fit(X_train, y_event_train, y_time_train, X_val=..., y_event_val=..., y_time_val=...)
```

### 5. Evaluation (`hybrid_survival.evaluation.metrics`)

```python
from hybrid_survival.evaluation.metrics import ModelEvaluator, SurvivalMetrics, print_model_comparison_table

evaluator = ModelEvaluator()
results = evaluator.evaluate_model(model, X_test, y_time_test, y_event_test, "Cox", eval_times=[2.4, 4.8, 9.7])
```

## Output Files

After running the pipeline, the following outputs are generated:

```
./models/
├── cox_model.pkl
├── deepsurv_model.pkl
├── struct_preprocessor.pkl
├── fusion_module.pkl
├── text_pca.pkl               # optional; PCA on text branch when used
└── embedder_config.pkl

./results/
├── model_comparison.csv       # Performance comparison
├── cox_calibration.png        # Cox calibration plot
├── deepsurv_calibration.png   # DeepSurv calibration plot
├── cox_survival_curves.png    # Cox survival curves
├── deepsurv_survival_curves.png
└── deepsurv_training.png      # Training history
```

## Performance Metrics

The pipeline reports:

1. **C-index (Concordance Index)**: Measures discriminative ability (0.5 = random, 1.0 = perfect)
2. **Brier Score**: Measures calibration at specific time points (lower is better)
3. **Event Rate**: Proportion of events in the dataset
4. **Calibration**: Agreement between predicted and observed probabilities

## Expected Results

On MIMIC-III mortality prediction:
- **Cox Model C-index**: ~0.70-0.75
- **DeepSurv C-index**: ~0.72-0.78
- **Improvement with text**: +2-5% over structured-only models

## Customization

### Using Different LLMs

```python
# In config.yaml, change:
model:
  llm_model: "emilyalsentzer/Bio_ClinicalBERT"  # Default
  # Or try:
  # llm_model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
  # llm_model: "allenai/scibert_scivocab_uncased"
```

### Adding Custom Features

```python
# Modify feature lists in main_pipeline.py
numeric_features = [
    'age', 'heart_rate', 'systolic_bp',
    'your_custom_feature_1',
    'your_custom_feature_2'
]
```

### Adjusting Model Architecture

```python
# In config.yaml:
survival:
  deepsurv:
    hidden_layers: [256, 128, 64, 32]  # Deeper network
    dropout: 0.4                        # More regularization
    learning_rate: 0.0005              # Lower learning rate
```

## Troubleshooting

### Common Issues

1. **Out of Memory (GPU)**
   - Reduce `batch_size` in config.yaml
   - Reduce `text_dim_reduction` in fusion settings
   - Use CPU instead: set `device: 'cpu'`

2. **Slow Training**
   - Reduce number of epochs
   - Reduce `max_sequence_length` for text processing
   - Use GPU if available

3. **Poor Performance**
   - Check data quality and missing values
   - Increase model complexity
   - Add more features
   - Perform hyperparameter tuning

## Citation

If you use this code, please cite:

```bibtex
@misc{hybrid-llm-survival,
  title={Hybrid LLM-Survival Model for Early Patient Risk Stratification},
  author={Shaima Rauf},
  year={2025},
  institution={NMAM Institute of Technology}
}
```

## References

1. Khader et al. (2023). Medical transformer for multimodal survival prediction
2. Hu et al. (2021). Transformer-Based Deep Survival Analysis
3. Wiegrebe et al. (2024). Deep learning for survival analysis: A review
4. Alsentzer et al. (2019). Publicly Available Clinical BERT Embeddings

## License

This project is for academic research purposes. See LICENSE file for details.

## Contact

For questions or issues:
- Email: shaima.rauf@example.com
- Project Guide: Dr. Vijay Murari
- Institution: NMAM Institute of Technology, Nitte

## Acknowledgments

- NMAM Institute of Technology for research support
- MIMIC-III dataset creators and maintainers
- HuggingFace for pre-trained ClinicalBERT models
