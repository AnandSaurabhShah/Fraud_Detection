import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st  # Import streamlit
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import defaultdict, deque

# ----------------------------
# (Re)define your model code
# ----------------------------

# Dataset file path (update if necessary)
DATASET_PATH = Path("C:/Users/Anand Shah/SPIT_HACK/backend/dataset.csv")

def time_to_minutes(time_str: str) -> float:
    """Convert a 'HH:MM:SS' timestamp to minutes."""
    hours, minutes, seconds = map(int, time_str.split(':'))
    return hours * 60 + minutes + seconds / 60

def engineer_features(df: pd.DataFrame, for_training: bool = False) -> pd.DataFrame:
    """Perform feature engineering on the DataFrame."""
    df["timestamp_minutes"] = df["timestamp"].apply(time_to_minutes)
    df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")
    df["from_id"] = pd.to_numeric(df["from_id"], errors="coerce")
    df["to_id"] = pd.to_numeric(df["to_id"], errors="coerce")
    df.fillna(0, inplace=True)
    if for_training and "is_fraud" in df.columns:
        return df
    return df[["timestamp_minutes", "transaction_amount", "from_id", "to_id"]]

class FraudDetector:
    """Fraud detection system using heuristics and an XGBoost classifier."""
    def __init__(self):
        self.model = XGBClassifier(scale_pos_weight=10, eval_metric="aucpr")
        self.user_history = defaultdict(deque)
        self.time_window = 60  # minutes

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=2)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        self.model.fit(X_resampled, y_resampled)

    def predict(self, X: np.ndarray, transaction: dict) -> int:
        # Update transaction timestamp
        transaction["timestamp_minutes"] = time_to_minutes(transaction["timestamp"])
        from_id = transaction["from_id"]
        timestamp = transaction["timestamp_minutes"]
        amount = transaction["transaction_amount"]

        if amount > 5000:
            return 1

        history = self.user_history[from_id]
        while history and (timestamp - history[0][0] > self.time_window):
            history.popleft()
        history.append((timestamp, amount))
        if sum(entry[1] for entry in history) >= 3500:
            return 1

        probability = self.model.predict_proba(X)[:, 1][0]
        return int(probability > 0.5)

def load_and_train_detector(dataset_path: Path) -> FraudDetector:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    
    training_data = pd.read_csv(str(dataset_path))
    training_df = engineer_features(training_data, for_training=True)
    X_train = training_df[["timestamp_minutes", "transaction_amount", "from_id", "to_id"]].values
    y_train = training_df["is_fraud"].values
    
    detector = FraudDetector()
    detector.train(X_train, y_train)
    return detector

def generate_random_transaction() -> dict:
    timestamp = f"{np.random.randint(0, 24):02}:{np.random.randint(0, 60):02}:{np.random.randint(0, 60):02}"
    return {
        "timestamp": timestamp,
        "transaction_amount": np.random.randint(1, 10000),
        "from_id": np.random.randint(1, 100),
        "to_id": np.random.randint(1, 100)
    }

# ----------------------------
# Streamlit App Code
# ----------------------------

st.title("Fraud Detection Model")
st.write("Enter transaction details or generate a random transaction to see if it is fraudulent.")

# Sidebar or main area inputs for a new transaction
with st.sidebar:
    st.header("Input Transaction Details")
    use_random = st.checkbox("Generate random transaction", value=True)
    
    if not use_random:
        timestamp_input = st.text_input("Timestamp (HH:MM:SS)", value="12:00:00")
        amount_input = st.number_input("Transaction Amount", min_value=1, max_value=10000, value=1000)
        from_id_input = st.number_input("From ID", min_value=1, max_value=100, value=50)
        to_id_input = st.number_input("To ID", min_value=1, max_value=100, value=50)

# Load and cache the detector so it isn’t retrained on every interaction.
@st.cache_resource(show_spinner=False)
def get_detector():
    return load_and_train_detector(DATASET_PATH)

detector = get_detector()

# Create a transaction dictionary
if use_random:
    txn = generate_random_transaction()
else:
    txn = {
        "timestamp": timestamp_input,
        "transaction_amount": amount_input,
        "from_id": from_id_input,
        "to_id": to_id_input
    }

st.subheader("Transaction Details")
st.write(txn)

# Prepare transaction data for prediction
txn_df = engineer_features(pd.DataFrame([txn]))
fraud_flag = detector.predict(txn_df.values, txn)
status = "Fraudulent" if fraud_flag == 1 else "Legitimate"

st.subheader("Prediction")
st.write(f"**Timestamp:** {txn['timestamp']} | **Fraud Status:** {status}")

# Option to run again (Streamlit re-runs the script on any interaction)
st.button("Re-run with new random transaction" if use_random else "Submit Transaction")
