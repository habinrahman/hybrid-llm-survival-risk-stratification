# Project Structure

```
hybrid-llm-survival/
│
├── config.yaml                    # Configuration file for all parameters
├── requirements.txt               # Python dependencies
├── README.md                      # Comprehensive documentation
├── run_pipeline.py               # Quick start script
│
├── Core Modules:
│   ├── data_preprocessing.py     # Structured data preprocessing
│   ├── text_embeddings.py        # ClinicalBERT embedding generation
│   ├── feature_fusion.py         # Multimodal feature fusion
│   ├── survival_models.py        # Cox and DeepSurv models
│   ├── model_evaluation.py       # Evaluation metrics and visualization
│   └── main_pipeline.py          # End-to-end pipeline orchestration
│
├── Notebooks:
│   └── demo_notebook.ipynb       # Interactive Jupyter notebook demo
│
├── data/                         # Dataset directory (MIMIC-III)
│   ├── admissions.csv
│   ├── patients.csv
│   ├── noteevents.csv
│   └── ... (other MIMIC-III tables)
│
├── models/                       # Saved models
│   ├── cox_model.pkl
│   ├── deepsurv_model.pkl
│   ├── struct_preprocessor.pkl
│   └── fusion_module.pkl
│
├── results/                      # Evaluation results and plots
│   ├── model_comparison.csv
│   ├── cox_calibration.png
│   ├── deepsurv_calibration.png
│   ├── cox_survival_curves.png
│   ├── deepsurv_survival_curves.png
│   └── deepsurv_training.png
│
└── logs/                         # Training logs
    └── training.log
```

## Module Descriptions

### 1. data_preprocessing.py
**Purpose:** Load and preprocess structured EHR data
**Key Classes:**
- `MIMICDataLoader`: Load MIMIC-III dataset and construct cohorts
- `StructuredDataPreprocessor`: Handle imputation, normalization, encoding

**Main Functions:**
- Missing value imputation (median, mean, mode)
- Feature normalization (standardization, min-max)
- Categorical encoding (label encoding)
- Data validation and cleaning

### 2. text_embeddings.py
**Purpose:** Generate embeddings from clinical notes
**Key Classes:**
- `ClinicalBERTEmbedder`: Wrapper for ClinicalBERT model
- `TextPreprocessor`: Clean and prepare clinical text

**Main Functions:**
- Text tokenization with ClinicalBERT tokenizer
- Batch processing for efficiency
- Multiple pooling strategies (CLS, mean, max)
- GPU acceleration support

### 3. feature_fusion.py
**Purpose:** Combine structured and text features
**Key Classes:**
- `FeatureFusion`: Multimodal feature fusion module
- `MultimodalDataset`: Container for multimodal patient data

**Fusion Methods:**
- Concatenation (default)
- Weighted combination
- Dimension reduction with PCA
- Feature normalization

### 4. survival_models.py
**Purpose:** Implement survival analysis models
**Key Classes:**
- `CoxModel`: Cox Proportional Hazards model wrapper
- `DeepSurvModel`: Deep learning survival network
- `DeepSurvNet`: PyTorch neural network architecture

**Model Features:**
- Cox: Classical statistical approach with regularization
- DeepSurv: Non-linear modeling with deep networks
- Risk score prediction
- Survival function estimation

### 5. model_evaluation.py
**Purpose:** Evaluate and compare models
**Key Classes:**
- `SurvivalMetrics`: Survival-specific metrics
- `ModelEvaluator`: Comprehensive evaluation framework

**Metrics:**
- Concordance Index (C-index)
- Brier Score
- Integrated Brier Score
- Calibration curves
- Cross-validation

**Visualizations:**
- Calibration plots
- Survival curves
- Training history
- Feature importance

### 6. main_pipeline.py
**Purpose:** Orchestrate the complete workflow
**Key Class:**
- `HybridSurvivalPipeline`: End-to-end pipeline manager

**Pipeline Steps:**
1. Load data
2. Preprocess structured features
3. Generate text embeddings
4. Fuse features
5. Split data
6. Train models
7. Evaluate models
8. Generate visualizations
9. Save results

## Configuration (config.yaml)

```yaml
model:
  llm_model: ClinicalBERT model path
  max_sequence_length: Token limit
  batch_size: Processing batch size

survival:
  cox: Cox model hyperparameters
  deepsurv: DeepSurv architecture and training

data:
  test_size: Train/test split ratio
  val_size: Validation split ratio
  imputation_strategy: How to handle missing values
  normalization: Feature scaling method

features:
  structured_features: List of EHR features
  lab_features: Laboratory measurements

outcomes:
  target: Prediction target (mortality/readmission)
  time_to_event: Time column name

paths:
  data_dir: Input data location
  models_dir: Model save location
  results_dir: Results save location
```

## Data Flow

```
MIMIC-III Raw Data
        ↓
    Data Loader
        ↓
    ┌───────────────────┐
    │ Structured Data   │   Clinical Notes
    │ (Demographics,    │   (Free Text)
    │  Vitals, Labs)    │
    └────────┬──────────┘        │
             ↓                   ↓
      Preprocessing      ClinicalBERT
             ↓                   ↓
      Feature Matrix      Embeddings (768-dim)
             └─────────┬─────────┘
                       ↓
                Feature Fusion
                  (PCA + Concat)
                       ↓
                 Fused Features
                       ↓
            ┌──────────┴──────────┐
            ↓                     ↓
        Cox Model            DeepSurv
            ↓                     ↓
     Risk Scores          Risk Scores
            └──────────┬──────────┘
                       ↓
                  Evaluation
              (C-index, Brier, etc.)
                       ↓
                  Visualizations
```

## Usage Patterns

### Quick Start
```bash
python run_pipeline.py
```

### Custom Configuration
```python
from main_pipeline import HybridSurvivalPipeline

pipeline = HybridSurvivalPipeline('custom_config.yaml')
pipeline.run_full_pipeline()
```

### Step-by-Step
```python
pipeline = HybridSurvivalPipeline()
pipeline.load_data()
pipeline.preprocess_structured_data()
# ... continue with other steps
```

### Using Saved Models
```python
import pickle

# Load model
with open('models/cox_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict
risk_scores = model.predict_risk(X_new)
```

## Output Files

### Models (./models/)
- Binary pickle files containing trained models
- Can be loaded for inference on new patients
- Include preprocessors for consistent transformation

### Results (./results/)
- CSV files with performance metrics
- PNG visualizations (300 DPI)
- Calibration and survival curve plots

### Logs (./logs/)
- Training progress logs
- Error messages and debugging info
- Timestamp-based log files

## Development Guidelines

### Adding New Features
1. Update feature lists in config.yaml
2. Ensure features exist in dataframe
3. Handle missing values appropriately
4. Retrain models with new features

### Experimenting with Models
1. Modify hyperparameters in config.yaml
2. Or create custom model instances
3. Use cross-validation for robust evaluation
4. Compare with baselines

### Using Different LLMs
1. Change `llm_model` in config.yaml
2. Ensure model is compatible with transformers
3. Adjust `max_sequence_length` if needed
4. May need to adjust `text_dim_reduction`

## Performance Optimization

### For Speed:
- Reduce batch size for text processing
- Use GPU if available
- Reduce number of epochs
- Use smaller hidden layers

### For Accuracy:
- Increase model complexity
- Add more features
- Use larger embeddings (reduce less)
- Perform hyperparameter tuning
- Use cross-validation

### Memory Management:
- Process data in batches
- Reduce text embedding dimension
- Use data generators for large datasets
- Clear GPU cache between runs
