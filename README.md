# NASA Vibration Anomaly Detection (Toward LSTM Autoencoder)

## Project Overview
This project focuses on building a machine learning pipeline capable of detecting abnormal vibration patterns in industrial machinery. The long-term objective is to develop an **LSTM autoencoder** for time-series anomaly detection to support predictive maintenance and reduce unplanned equipment downtime.

Rather than jumping directly into deep learning, this project follows a professional modeling progression:

**Baseline → Data Understanding → Sequence Modeling → Deep Learning**

---

## Current Status (Proof of Concept)

✅ Data ingestion from the IMS bearing dataset  
✅ Time-series preprocessing and normalization  
✅ Sequence generation for temporal modeling  
✅ Isolation Forest baseline for unsupervised anomaly detection  
✅ Memory-efficient batching for large datasets  
🔄 Transitioning toward LSTM autoencoder architecture  

This proof of concept establishes that anomalies can be detected in vibration data and provides a strong foundation for more advanced models.

---

## Why This Project Matters
Mechanical failures are expensive and disruptive. Detecting abnormal vibration patterns early allows maintenance teams to intervene before catastrophic failure occurs.

Potential impact:

- Reduced operational downtime  
- Lower maintenance costs  
- Improved equipment lifespan  
- Increased safety  

---

## Technical Approach

### 1. Data Pipeline
- Load raw vibration sensor files  
- Normalize signals using `StandardScaler`  
- Convert continuous signals into fixed-length sequences  
- Batch process large datasets to prevent memory exhaustion  

### 2. Baseline Model
An **Isolation Forest** is used as the initial anomaly detector because:

- No labeled failure data is required  
- It performs well for high-dimensional outlier detection  
- It provides a fast benchmark before deep learning  

### 3. Next Phase — LSTM Autoencoder
The next step is implementing an LSTM autoencoder to learn normal temporal behavior and detect deviations via reconstruction error.

This is expected to improve detection of subtle, time-dependent failure patterns that tree-based models may miss.

---

## Tech Stack
- Python  
- NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  

**Planned:**
- PyTorch or TensorFlow  
- Model checkpointing  
- Evaluation metrics  
- Experiment tracking  

---

## Key Engineering Lessons So Far
- Handling large time-series datasets requires batching strategies.
- Memory constraints must be considered early in pipeline design.
- Establishing a baseline model is critical before increasing model complexity.
- Visualization at scale requires thoughtful sampling/windowing.

---

## Future Improvements
- Implement LSTM autoencoder  
- Compare deep learning vs tree-based anomaly detection  
- Add evaluation methodology  
- Refactor notebook code into modular scripts  
- Explore real-time inference potential  

---
