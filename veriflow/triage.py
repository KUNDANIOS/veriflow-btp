"""
VeriFlow Triage Engine
======================
Confidence-calibrated triage layer that sits between a raw AI violation
detector and the challan-issuance step.

Given a set of features about a flagged violation, it produces:
  - calibrated_confidence  (0.0 – 1.0)
  - triage_decision        ("auto_clear" | "human_review" | "auto_reject")
  - evidence_notes         (list of strings explaining the decision)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pickle, os, json
from datetime import datetime

# ── Thresholds (tune these to trade off queue reduction vs accuracy) ──────────
AUTO_CLEAR_THRESHOLD  = 0.80   # confidence >= this → auto-clear (challan issued)
AUTO_REJECT_THRESHOLD = 0.25   # confidence <  this → auto-reject (no challan)
# Everything in between goes to human review


def _extract_hour(dt_str):
    """Parse hour from ISO datetime string."""
    try:
        return int(str(dt_str)[11:13])
    except Exception:
        return -1


def _is_peak_hour(hour):
    """Bengaluru peak hours: 8–10 AM and 5–8 PM."""
    return int(hour in list(range(8, 11)) + list(range(17, 21)))


def _is_night(hour):
    return int(hour >= 22 or hour <= 5)


VIOLATION_RISK = {
    "WRONG PARKING": 0.65,
    "NO PARKING": 0.72,
    "PARKING IN A MAIN ROAD": 0.68,
    "PARKING ON FOOTPATH": 0.75,
    "DEFECTIVE NUMBER PLATE": 0.55,
    "HELMET": 0.85,
    "TRIPLE RIDING": 0.82,
    "SEATBELT": 0.70,
    "SIGNAL JUMP": 0.88,
    "WRONG SIDE": 0.80,
}

VEHICLE_RISK = {
    "SCOOTER": 0.6,
    "MOTOR CYCLE": 0.65,
    "CAR": 0.55,
    "PASSENGER AUTO": 0.6,
    "MAXI-CAB": 0.5,
    "LGV": 0.45,
    "GOODS AUTO": 0.5,
    "PRIVATE BUS": 0.45,
    "MOPED": 0.6,
    "VAN": 0.5,
    "UNKNOWN": 0.5,
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw BTP violation records.
    Works with the provided violations CSV schema.
    """
    feat = pd.DataFrame()

    # Time features
    feat["hour"] = df["created_datetime"].apply(_extract_hour)
    feat["is_peak_hour"] = feat["hour"].apply(_is_peak_hour)
    feat["is_night"] = feat["hour"].apply(_is_night)

    # Violation complexity (more violation types in one flag → more ambiguous)
    def count_violations(v):
        try:
            return len(json.loads(v.replace("'", '"')))
        except Exception:
            return 1
    feat["num_violations"] = df["violation_type"].apply(count_violations)

    # Violation risk score
    def viol_risk(v):
        try:
            types = json.loads(v.replace("'", '"'))
            return max(VIOLATION_RISK.get(t.upper(), 0.5) for t in types)
        except Exception:
            return 0.5
    feat["violation_risk"] = df["violation_type"].apply(viol_risk)

    # Vehicle risk
    feat["vehicle_risk"] = df["vehicle_type"].str.upper().map(
        lambda v: VEHICLE_RISK.get(v, 0.5)
    )

    # Has junction info (camera-based → more reliable)
    feat["has_junction"] = (
        df["junction_name"].apply(lambda x: 0 if str(x).strip() in ["No Junction", "NULL", "nan", ""] else 1)
    )

    # Data sent to SCITA (integrated system → more reliable)
    feat["sent_to_scita"] = df["data_sent_to_scita"].apply(
        lambda x: 1 if str(x).strip().upper() == "TRUE" else 0
    )

    # Number plate was updated after initial flag (indicates review caught an issue)
    feat["plate_updated"] = df.apply(
        lambda r: 0 if (
            str(r.get("updated_vehicle_number", "")).strip() in ["", "NULL", "nan"] or
            r.get("updated_vehicle_number") == r.get("vehicle_number")
        ) else 1,
        axis=1
    )

    # Police station encoded as rejection-rate proxy
    # (computed later if training; else 0.5 default)
    feat["station_risk"] = 0.5

    return feat


def compute_station_risk(df: pd.DataFrame) -> dict:
    """
    Compute per-station historical rejection rate from training data.
    Higher rejection rate → AI flags from that station are less reliable.
    """
    validated = df[df["validation_status"].isin(["approved", "rejected"])].copy()
    station_stats = validated.groupby("police_station")["validation_status"].apply(
        lambda x: (x == "rejected").sum() / len(x)
    ).to_dict()
    return station_stats


class VeriFlowTriageEngine:
    """
    Trains on historical BTP violation records and provides calibrated
    confidence scores for new flagged violations.
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self.station_risk_map = {}
        self.label_encoder = LabelEncoder()
        self.model_path = model_path or "models/veriflow_model.pkl"
        self.metrics = {}

    def train(self, df: pd.DataFrame):
        """Train on historical validated records."""
        print(f"[VeriFlow] Training on {len(df):,} records...")

        # Only use records with a clear human decision
        labelled = df[df["validation_status"].isin(["approved", "rejected"])].copy()
        print(f"[VeriFlow] Labelled records: {len(labelled):,} "
              f"({(labelled['validation_status']=='approved').sum():,} approved, "
              f"{(labelled['validation_status']=='rejected').sum():,} rejected)")

        # Compute station risk
        self.station_risk_map = compute_station_risk(df)

        # Build features
        X = build_features(labelled)
        X["station_risk"] = labelled["police_station"].map(
            lambda s: self.station_risk_map.get(s, 0.30)
        )

        y = (labelled["validation_status"] == "approved").astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        base_model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        )
        self.model = CalibratedClassifierCV(base_model, cv=5, method="isotonic")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
            "f1":        round(f1_score(y_test, y_pred), 4),
            "test_size": len(y_test),
        }

        # Simulate triage on test set
        triage_results = self._simulate_triage(y_prob, y_test.values)
        self.metrics.update(triage_results)

        print(f"[VeriFlow] Model trained. Accuracy={self.metrics['accuracy']}, F1={self.metrics['f1']}")
        print(f"[VeriFlow] Triage: auto_clear={triage_results['auto_clear_pct']:.1f}%, "
              f"human_review={triage_results['human_review_pct']:.1f}%, "
              f"auto_reject={triage_results['auto_reject_pct']:.1f}%")
        print(f"[VeriFlow] Queue reduction: {triage_results['queue_reduction_pct']:.1f}%")

        self.save()
        return self.metrics

    def _simulate_triage(self, probs, labels):
        """Simulate the triage split and compute key metrics."""
        n = len(probs)
        auto_clear  = probs >= AUTO_CLEAR_THRESHOLD
        auto_reject = probs < AUTO_REJECT_THRESHOLD
        human_rev   = ~auto_clear & ~auto_reject

        # Accuracy of auto decisions
        ac_correct = np.mean(labels[auto_clear] == 1) if auto_clear.sum() > 0 else 0
        ar_correct = np.mean(labels[auto_reject] == 0) if auto_reject.sum() > 0 else 0

        queue_reduction = (auto_clear.sum() + auto_reject.sum()) / n * 100

        return {
            "auto_clear_count": int(auto_clear.sum()),
            "auto_reject_count": int(auto_reject.sum()),
            "human_review_count": int(human_rev.sum()),
            "auto_clear_pct": auto_clear.sum() / n * 100,
            "auto_reject_pct": auto_reject.sum() / n * 100,
            "human_review_pct": human_rev.sum() / n * 100,
            "auto_clear_accuracy": round(float(ac_correct), 4),
            "auto_reject_accuracy": round(float(ar_correct), 4),
            "queue_reduction_pct": queue_reduction,
        }

    def predict(self, features: pd.DataFrame, police_station: str = "UNKNOWN"):
        """
        Score a single or batch of violation features.
        Returns a list of dicts with confidence + triage decision.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        features = features.copy()
        features["station_risk"] = self.station_risk_map.get(police_station, 0.30)

        probs = self.model.predict_proba(features)[:, 1]
        results = []
        for p in probs:
            if p >= AUTO_CLEAR_THRESHOLD:
                decision = "auto_clear"
                label = "✅ Auto-Clear"
                color = "green"
            elif p < AUTO_REJECT_THRESHOLD:
                decision = "auto_reject"
                label = "❌ Auto-Reject"
                color = "red"
            else:
                decision = "human_review"
                label = "🔍 Human Review"
                color = "orange"

            results.append({
                "confidence": round(float(p), 4),
                "triage_decision": decision,
                "triage_label": label,
                "triage_color": color,
                "evidence_notes": self._generate_evidence_notes(features.iloc[0], p),
            })
        return results

    def _generate_evidence_notes(self, feat_row, confidence):
        notes = []
        if feat_row.get("has_junction", 0) == 1:
            notes.append("Camera at known junction — higher spatial reliability")
        if feat_row.get("sent_to_scita", 0) == 1:
            notes.append("Record integrated with SCITA — cross-validated")
        if feat_row.get("is_peak_hour", 0) == 1:
            notes.append("Flagged during peak hours — context consistent")
        if feat_row.get("is_night", 0) == 1:
            notes.append("Night-time flag — lighting may reduce reliability")
        if feat_row.get("num_violations", 1) > 2:
            notes.append("Multiple violation types in one flag — review recommended")
        if feat_row.get("plate_updated", 0) == 1:
            notes.append("Vehicle plate was updated post-flag — ANPR discrepancy noted")
        station_risk = feat_row.get("station_risk", 0.3)
        if station_risk > 0.35:
            notes.append(f"Station has elevated historical rejection rate ({station_risk*100:.0f}%) — route to review")
        if confidence >= AUTO_CLEAR_THRESHOLD:
            notes.append(f"Confidence {confidence:.2f} ≥ {AUTO_CLEAR_THRESHOLD} threshold — auto-cleared")
        elif confidence < AUTO_REJECT_THRESHOLD:
            notes.append(f"Confidence {confidence:.2f} < {AUTO_REJECT_THRESHOLD} threshold — auto-rejected")
        else:
            notes.append(f"Confidence {confidence:.2f} in ambiguous range — routed to human reviewer")
        return notes

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "station_risk_map": self.station_risk_map,
                "metrics": self.metrics,
            }, f)
        print(f"[VeriFlow] Model saved to {self.model_path}")

    def load(self):
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.station_risk_map = data["station_risk_map"]
        self.metrics = data["metrics"]
        print(f"[VeriFlow] Model loaded from {self.model_path}")
        return self
