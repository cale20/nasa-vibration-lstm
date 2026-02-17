# NASA Vibration LSTM – Industrial Anomaly Detection

Predictive maintenance with vibration sensor data from NASA’s IMS dataset.

This project demonstrates ML pipeline design for anomaly detection in rotating machinery, laying the groundwork for a deep learning LSTM autoencoder.

---

## 🚀 Highlights / Skills Demonstrated

- **End-to-end preprocessing pipeline**
  - Automatic file discovery and filtering
  - Global scaling for consistent anomaly detection
  - Disk-backed sequence dataset (np.memmap) for millions of sequences
- **Baseline anomaly detection**
  - Isolation Forest trained on healthy machine data
  - Generation of Machine Health Curve for temporal anomaly trends
- **Professional ML practices**
  - Config-driven design for reproducibility
  - Modular `src/` structure (preprocessing, training, utils)
  - Separation of exploratory notebooks and production-ready code
- **Visualization & Analysis**
  - Visual inspection of vibration signals
  - Mean anomaly scores per file to track machine degradation

---

## 📂 Project Structure (Simplified)
```text
nasa-vibration-lstm/
├── data
│   ├── processed
│   │   ├── figures
│   │   ├── all_sequences.dat
│   │   ├── all_sequences.dat.meta.json
│   │   ├── anomaly_scores.npy
│   │   ├── global_scaler.save
│   │   └── isolation_forest.model
│   └── raw
│       └── IMS
├── models
├── notebooks
│   └── baseline_anomaly.ipynb
├── src
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── run_preprocessing.py
│   ├── train_isolation_forest.py
│   ├── utils.py
│   └── __init__.py
├── tests
├── .gitignore
├── README.md
└── requirements.txt

```

---

## ⚡ Quick Start (Demo-ready)

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

How to run:

```powershell
python -m src.pipeline
```

Machine health curve figure is saved in `data/processed/figures/`

## 📈 Key Takeaways

- **Global scaling** prevents anomalies from being normalized away, ensuring the model can detect deviations reliably.  
- **Disk-backed datasets (np.memmap)** allow large-scale experimentation on millions of sequences without exceeding RAM limits.  
- **Baseline models like Isolation Forest** provide sanity checks before building complex deep learning models.  
- **Clean modular code, config-driven pipelines, and reproducible preprocessing** are professional ML practices that make your pipeline maintainable and production-ready.  

---

## 🔜 Next Steps

- Implement PyTorch `Dataset` / `DataLoader` for LSTM autoencoder training  
- Build a dense autoencoder for intermediate experiments  
- Train an LSTM autoencoder for temporal anomaly detection  
- Evaluate predictive performance and determine anomaly thresholds  

---

## 🏆 Outcome So Far

- Successfully processed **>13 million vibration sequences**  
- Trained **Isolation Forest baseline** on healthy data  
- Generated **Machine Health Curve** for temporal anomaly monitoring  

---

## 📚 References

- [NASA IMS Bearing Dataset](https://data.nasa.gov/dataset/ims-bearings)  
- [Isolation Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)  
- [NumPy Memmap Documentation](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)  






