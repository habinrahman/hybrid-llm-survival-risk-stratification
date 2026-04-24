# Getting Started with Hybrid LLM-Survival Model

## Welcome! 🎉

This guide will help you get your Hybrid LLM-Survival Model up and running quickly.

## What You've Built

A complete end-to-end system for patient risk stratification that:
- ✅ Processes structured EHR data (demographics, vitals, labs)
- ✅ Encodes clinical notes using ClinicalBERT (state-of-the-art medical NLP)
- ✅ Fuses multimodal features intelligently
- ✅ Trains both classical (Cox) and deep learning (DeepSurv) survival models
- ✅ Provides comprehensive evaluation and visualization
- ✅ Delivers interpretable risk predictions for clinical use

## Quick Start (3 Minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch (deep learning)
- Transformers (ClinicalBERT)
- lifelines (survival analysis)
- scikit-learn (preprocessing)
- matplotlib/seaborn (visualization)
- And other essential libraries

### 2. Run the Complete Pipeline

```bash
python run_pipeline.py
```

This single command will:
1. Load sample MIMIC-III data
2. Preprocess structured features
3. Generate ClinicalBERT embeddings
4. Fuse multimodal features
5. Train Cox and DeepSurv models
6. Evaluate and compare models
7. Generate visualizations
8. Save everything to disk

**Expected runtime:** 10-15 minutes on CPU, 3-5 minutes on GPU

### 3. Explore Results

After completion, check:
- `./results/model_comparison.csv` - Performance metrics
- `./results/*.png` - Visualizations
- `./models/*.pkl` - Trained models
- Open `demo_notebook.ipynb` for interactive exploration

## Step-by-Step Tutorial

### Option 1: Interactive Notebook (Recommended for Learning)

```bash
jupyter notebook demo_notebook.ipynb
```

The notebook walks through each step with:
- Detailed explanations
- Visualizations at each stage
- Sample predictions
- Model interpretation

### Option 2: Python Scripts (Recommended for Production)

```python
from main_pipeline import HybridSurvivalPipeline

# Initialize
pipeline = HybridSurvivalPipeline('config.yaml')

# Run complete pipeline
results = pipeline.run_full_pipeline()

# Or run step-by-step
pipeline.load_data(outcome='mortality')
pipeline.preprocess_structured_data()
pipeline.generate_text_embeddings()
pipeline.fuse_features()
pipeline.split_data()
pipeline.train_models()
pipeline.evaluate_models()
pipeline.visualize_results()
pipeline.save_models()
```

## Customization Guide

### 1. Use Your Own Data

**Requirements:**
- CSV file with patient records
- Required columns: `subject_id`, `clinical_notes`, `event`, `time`
- Structured features: demographics, vitals, labs

**Steps:**
1. Place your data in `./data/` directory
2. Update `MIMICDataLoader` in `data_preprocessing.py` to load your data
3. Or modify the loader's `load_cohort()` method

**Example:**
```python
# In data_preprocessing.py
def load_cohort(self):
    # Load your custom data
    df = pd.read_csv('path/to/your/data.csv')
    return df
```

### 2. Change the LLM Model

**In `config.yaml`:**
```yaml
model:
  llm_model: "emilyalsentzer/Bio_ClinicalBERT"  # Current
  # Try these alternatives:
  # llm_model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
  # llm_model: "allenai/biomed_roberta_base"
```

### 3. Adjust Model Architecture

**For DeepSurv (in `config.yaml`):**
```yaml
survival:
  deepsurv:
    hidden_layers: [128, 64, 32]  # Change to [256, 128, 64] for deeper network
    dropout: 0.3                   # Increase to 0.4 for more regularization
    learning_rate: 0.001           # Decrease to 0.0005 for more stable training
    epochs: 100                    # Increase to 200 for better convergence
```

**For Cox (in `config.yaml`):**
```yaml
survival:
  cox:
    penalizer: 0.01  # Increase for more regularization
    l1_ratio: 0.0    # Set to 0.5 for L1+L2 regularization
```

### 4. Add More Features

**In `main_pipeline.py`:**
```python
numeric_features = [
    'age', 'heart_rate', 'systolic_bp',
    # Add your new features here:
    'bmi', 'blood_pressure', 'oxygen_saturation'
]
```

Make sure these features exist in your dataframe!

### 5. Change Prediction Target

**In `config.yaml`:**
```yaml
outcomes:
  target: "mortality"        # Change to "readmission_30d"
  time_to_event: "los_days"  # Or "days_to_readmit"
```

## Understanding the Output

### 1. Model Comparison Table

```
Model      C-index  N Samples  N Events  Event Rate
Cox Model  0.7234   1000       312       31.20%
DeepSurv   0.7456   1000       312       31.20%
```

**Interpretation:**
- **C-index > 0.7**: Good discrimination
- **C-index > 0.8**: Excellent discrimination
- Higher is better (random = 0.5, perfect = 1.0)

### 2. Visualizations

**Calibration Plots:**
- Shows how well predicted probabilities match observed outcomes
- Points close to diagonal = well-calibrated
- Points above diagonal = underestimating risk
- Points below diagonal = overestimating risk

**Survival Curves:**
- Each line = predicted survival probability for one patient
- Steeper decline = higher risk
- Flatter curve = lower risk
- Use these to communicate risk to clinicians

**Training History:**
- Shows DeepSurv loss over epochs
- Should decrease and stabilize
- If not converging, increase epochs or decrease learning rate

### 3. Saved Models

```python
import pickle

# Load a trained model
with open('models/deepsurv_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions on new patient
new_patient_features = ...  # Preprocessed features
risk_score = model.predict_risk(new_patient_features)
survival_curve = model.predict_survival_function(new_patient_features, times=[1,7,30])
```

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory
**Solution:**
```yaml
# In config.yaml, reduce batch size:
model:
  batch_size: 8  # Down from 16
survival:
  deepsurv:
    batch_size: 32  # Down from 64
```

### Issue 2: Slow Training
**Solution 1 - Use GPU:**
```python
# Check if GPU is available
import torch
print(torch.cuda.is_available())  # Should be True
```

**Solution 2 - Reduce Complexity:**
```yaml
model:
  max_sequence_length: 256  # Down from 512
survival:
  deepsurv:
    epochs: 50  # Down from 100
    hidden_layers: [64, 32]  # Smaller network
```

### Issue 3: Poor Model Performance
**Try these:**

1. **Add more features:**
```python
numeric_features = [
    'age', 'heart_rate', 'systolic_bp',
    # Add more clinical measurements
]
```

2. **Increase model capacity:**
```yaml
survival:
  deepsurv:
    hidden_layers: [256, 128, 64, 32]  # Deeper
```

3. **Reduce text dimension reduction:**
```python
fusion = FeatureFusion(
    text_dim_reduction=128  # Keep more text information
)
```

4. **Use cross-validation:**
```python
from model_evaluation import cross_validate_survival

results = cross_validate_survival(
    DeepSurvModel,
    X, y_time, y_event,
    n_folds=5
)
```

### Issue 4: Missing Dependencies
**Solution:**
```bash
# Install specific versions if issues occur
pip install torch==2.0.0
pip install transformers==4.30.0
pip install lifelines==0.27.0
```

## Advanced Usage

### 1. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
# (Would need to implement GridSearchCV wrapper for survival models)

# Or manually:
learning_rates = [0.001, 0.0001]
hidden_layers_options = [[128, 64], [256, 128, 64]]

for lr in learning_rates:
    for hidden in hidden_layers_options:
        model = DeepSurvModel(
            input_dim=X_train.shape[1],
            hidden_layers=hidden,
            learning_rate=lr
        )
        # Train and evaluate...
```

### 2. Ensemble Models

```python
# Combine Cox and DeepSurv predictions
cox_risk = cox_model.predict_risk(X_test)
deepsurv_risk = deepsurv_model.predict_risk(X_test)

# Average risk scores
ensemble_risk = (cox_risk + deepsurv_risk) / 2

# Evaluate
c_index = SurvivalMetrics.concordance_index(
    y_time_test, ensemble_risk, y_event_test
)
```

### 3. Feature Importance Analysis

```python
# Get Cox coefficients
coef_df = cox_model.get_coefficients()
top_features = coef_df.nlargest(10, 'coef')

# Visualize
import matplotlib.pyplot as plt
top_features['coef'].plot(kind='barh')
plt.title('Top 10 Risk Factors')
plt.show()
```

### 4. Deploy for Production

```python
# Create inference function
def predict_patient_risk(patient_data):
    """
    Predict risk for a new patient
    
    Args:
        patient_data: Dict with patient features
    
    Returns:
        Risk score and survival probability
    """
    # Load preprocessors and model
    with open('models/struct_preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open('models/deepsurv_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Preprocess
    X = preprocessor.transform(patient_data)
    
    # Predict
    risk = model.predict_risk(X)
    survival_30d = model.predict_survival_function(X, times=[30])
    
    return {
        'risk_score': float(risk[0]),
        'survival_prob_30d': float(survival_30d[0, 0])
    }
```

## Next Steps

1. **Explore the Demo Notebook** - Best way to understand each component
2. **Experiment with Hyperparameters** - Use config.yaml for quick changes
3. **Add Your Own Data** - Replace synthetic data with real MIMIC-III
4. **Try Different LLMs** - Compare ClinicalBERT, BioBERT, PubMedBERT
5. **Perform Cross-Validation** - For robust performance estimates
6. **Deploy the Model** - Create an API for clinical decision support

## Resources

**Documentation:**
- README.md - Complete project documentation
- PROJECT_STRUCTURE.md - Detailed architecture
- Module docstrings - In-code documentation

**Learning:**
- demo_notebook.ipynb - Interactive tutorial
- Code comments - Throughout all modules

**References:**
- See references in your project report
- ClinicalBERT paper: Alsentzer et al. (2019)
- DeepSurv paper: Katzman et al. (2018)
- Cox model: Cox (1972)

## Support

For issues or questions:
1. Check this guide first
2. Review error messages carefully
3. Search for similar issues online
4. Contact your guide: Dr. Vijay Murari

## Conclusion

You now have a complete, production-ready system for patient risk stratification! 🎉

The system combines:
- ✅ State-of-the-art NLP (ClinicalBERT)
- ✅ Classical statistics (Cox model)
- ✅ Deep learning (DeepSurv)
- ✅ Comprehensive evaluation
- ✅ Clinical interpretability

**Good luck with your project!** 🚀

---

*Project: Hybrid LLM-Survival Model for Early Patient Risk Stratification*  
*Author: Shaima Rauf (NNM24CSE18)*  
*Institution: NMAM Institute of Technology, Nitte*  
*Date: February 2026*
