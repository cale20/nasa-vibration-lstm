# --------------------------------------------------
# train_isolation_forest.py
# --------------------------------------------------
"""
Train and save Isolation Forest
"""
import os
from sklearn.ensemble import IsolationForest
import joblib
import numpy as np
from .dataset import load_memmap_dataset
from .config import CONFIG

def train():
    X = load_memmap_dataset(flatten_for_tree=True)
    # Subset for fast training
    if X.shape[0] > CONFIG["max_train_samples"]:
        idx = np.random.choice(X.shape[0], CONFIG["max_train_samples"], replace=False)
        X_train = X[idx]
    else:
        X_train = X

    model = IsolationForest(
        n_estimators=CONFIG["n_estimators"],
        contamination=CONFIG["contamination"],
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    joblib.dump(model, os.path.join(CONFIG["processed_folder"], "isolation_forest.model"))
    print(f"✅ Isolation Forest trained | Training samples: {X_train.shape}")

if __name__ == "__main__":
    train()
