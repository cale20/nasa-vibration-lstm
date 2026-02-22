# NASA Vibration LSTM – Industrial Anomaly Detection

Predictive maintenance with vibration sensor data from NASA’s IMS dataset.

This project demonstrates ML pipeline design for anomaly detection in rotating machinery, laying the groundwork for a deep learning LSTM autoencoder.

---

## 🚀 Project Highlights

- **End-to-end preprocessing pipeline**
  - Automatic file discovery and filtering
  - Global scaling for consistent anomaly detection
  - Disk-backed sequence dataset (np.memmap) for millions of sequences
- **Baseline anomaly detection**
  - Isolation Forest trained on healthy machine data
  - Generation of Machine Health Curve for temporal anomaly trends
- **Engineering practices**
  - Config-driven design for reproducibility
  - Modular `src/` structure (preprocessing, training, utils)
  - Separation of exploratory notebooks and pipeline scripts
- **Visualization & Analysis**
  - Visual inspection of vibration signals
  - Mean anomaly scores per file to track machine degradation

---

## 📂 Project Structure (Simplified)
```text
├── data/
│ ├── raw/IMS/ # Original vibration files
│ └── processed/ # Scaler, split memmaps, diagnostics, trained artifacts
├── notebooks/ # Exploratory analysis & visualization
├── src/ # Production-ready ML pipeline
│ ├── config.py
│ ├── dataset.py
│ ├── preprocessing.py
│ ├── train_isolation_forest.py
│ ├── train_dense_autoencoder.py
│ ├── train_lstm_autoencoder.py
│ ├── evaluate.py
│ ├── evaluate_autoencoder.py
│ ├── compare_models.py
│ └── utils.py
├── models/ # Saved model checkpoints
├── tests/ # Unit and pipeline sanity checks
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start
Prerequisites:

- Python 3.11.9 (create and activate a virtual environment before installing)

Install runtime dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Data placement:

- Download [NASA IMS Bearing Dataset](https://data.nasa.gov/dataset/ims-bearings) and place it under `data/raw/`

Run preprocessing + baseline:

```powershell
python -m src.run_preprocessing
python -m src.train_isolation_forest
python -m src.evaluate
```

Run autoencoders:

```powershell
# Dense autoencoder
python -m src.train_dense_autoencoder
python -m src.evaluate_autoencoder --model-type dense

# LSTM autoencoder
python -m src.train_lstm_autoencoder
python -m src.evaluate_autoencoder --model-type lstm
```

Generate side-by-side model trend comparison:

```powershell
python -m src.compare_models
```

Main outputs are written under `data/processed/diagnostics/`.

## 📈 Technical Takeaways

- **Global scaling** preserves absolute signal shifts, which keeps anomalies detectable across the machine life cycle.  
- **Disk-backed datasets (`np.memmap`)** support large-scale experiments without requiring all sequences in RAM.  
- **Isolation Forest baseline results** provide a reference point before evaluating deeper sequence models.  
- **Modular, config-driven preprocessing and evaluation** improve reproducibility and simplify iteration.  

---

## 🧠 Data Split Strategy

- `healthy_train`: early-life healthy files used to fit anomaly models  
- `healthy_val`: healthy holdout files used for threshold selection  
- `test_mixed`: later-life files used for trend monitoring and anomaly-rate analysis

Split-aware artifacts produced during preprocessing:

- `all_sequences.dat`
- `healthy_train_sequences.dat`
- `healthy_val_sequences.dat`
- `split_metadata.json`

Note: in some Windows environments, the full `all_sequences.dat` allocation may be skipped due to file-mapping limits. In that case, evaluation automatically falls back to streaming from raw files using `split_metadata.json`.

## 📊 How to Interpret Outputs

- `isolation_forest_file_metrics.json`: per-file baseline mean score and anomaly rate  
- `dense_autoencoder_file_metrics.json` / `lstm_autoencoder_file_metrics.json`: per-file reconstruction trends  
- `*_threshold.json`: saved threshold and percentile rule used for anomaly decisions  
- `model_comparison_anomaly_rate.png`: normalized trend comparison across baseline and autoencoders

Threshold policy:

- Isolation Forest threshold is computed from `healthy_val` scores (leakage-safe).  
- Autoencoder thresholds are computed from `healthy_val` reconstruction error percentiles.

Practical reading pattern:

1. Confirm healthy period has lower anomaly rates than late-life period.  
2. Check that threshold is stable when retraining with the same split rule.  
3. Compare IF vs Dense AE vs LSTM AE trends, then tune hyperparameters.

## 🧪 Evaluation Scope

This project currently demonstrates unsupervised anomaly trend detection and model comparison on run-to-failure data.  
Formal change-window accuracy scoring is the next evaluation milestone.

---

## 🏆 Current Outcomes

- Successfully processed **>13 million vibration sequences**  
- Trained **Isolation Forest baseline** on healthy data  
- Generated **Machine Health Curve** for temporal anomaly monitoring  
- Added **Dense and LSTM autoencoder training/evaluation scripts**  
- Added **split-aware preprocessing + diagnostics + comparison tooling**

---

## 📚 References

- [NASA IMS Bearing Dataset](https://data.nasa.gov/dataset/ims-bearings)  
- [Isolation Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)  
- [NumPy Memmap Documentation](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)  







