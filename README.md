# 🚦 VeriFlow — AI-Verified Traffic Violation Detection System

> **Flipkart Gridlock Hackathon 2.0 · Prototype Round 2**
> Theme: Automated Photo Identification and Classification for Traffic Violations Using Computer Vision

---

## 🎯 What is VeriFlow?

Bengaluru Traffic Police (BTP) already operate AI cameras at 330+ junctions.
But **30.1% of AI-flagged violations are false positives** — and every single flag
is manually re-validated by TMC staff before a challan goes out.

**VeriFlow is not another violation detector.**

It is a **confidence-calibrated triage layer** that sits between the raw AI detector
and the challan-issuance step. It scores each flag and routes it into one of three lanes:

| Lane | Condition | Action |
|------|-----------|--------|
| ✅ **Auto-Clear** | Confidence ≥ 0.80 | Challan issued directly — no human needed |
| 🔍 **Human Review** | 0.25 ≤ Confidence < 0.80 | Routed to TMC reviewer with highlighted evidence |
| ❌ **Auto-Reject** | Confidence < 0.25 | Discarded + logged for model retraining |

**Result on real BTP data: 15%+ queue reduction with 84.5% auto-clear accuracy**

---

## 📊 Key Numbers (from real BTP dataset)

| Metric | Value |
|--------|-------|
| Total violation flags (Jan–May 2024) | 298,450 |
| Human-validated records | 165,154 |
| **False positive rate** | **30.1%** |
| Unnecessary reviews (wasted TMC time) | 49,754 |
| VeriFlow queue reduction | ~15–40% (tunable) |
| Auto-clear accuracy | 84.5% |
| Model F1 Score | 0.82 |

---

## 🏗️ Architecture

```
Camera / Body-Worn Cam / FTVR Image
           │
           ▼
   ┌──────────────┐
   │ Preprocessing │  enhance, denoise, deblur
   └──────┬───────┘
           ▼
   ┌──────────────┐
   │ YOLOv8       │  vehicle & person detection
   │ Detector     │  violation classification
   └──────┬───────┘
           ▼
   ┌────────────────────────────────┐
   │  VeriFlow Triage Engine        │  ← CORE INNOVATION
   │  - Multi-signal confidence     │
   │  - Station risk calibration    │
   │  - Calibrated GBM model        │
   └───────┬────────┬───────────────┘
           │        │
    ✅ Auto    🔍 Review   ❌ Reject
    Challan   TMC Queue   Log + Retrain
           │
           ▼
   ┌──────────────┐
   │ ANPR / OCR   │  plate recognition
   └──────┬───────┘
           ▼
   ┌──────────────┐
   │ Evidence &   │  annotated image, audit trail
   │ Analytics    │  dashboard, heatmap
   └──────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/veriflow.git
cd veriflow
pip install -r requirements.txt
```

### 2. Add the Dataset

Place the BTP violation CSV in `data/violations.csv`
(the file provided by HackerEarth: `jan_to_may_police_violation_anonymized.csv`)

### 3. Run the Dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501

---

## 📁 Project Structure

```
veriflow/
├── app.py                    # Streamlit dashboard (5 pages)
├── requirements.txt
├── data/
│   ├── violations.csv        # BTP parking violations dataset
│   └── events.csv            # BTP event data
├── veriflow/
│   ├── __init__.py
│   ├── triage.py             # VeriFlow confidence + triage engine
│   ├── analytics.py          # Charts, metrics, heatmaps
│   └── detector.py           # YOLOv8 CV pipeline + annotation
├── models/
│   └── veriflow_model.pkl    # Trained triage model (auto-generated)
└── notebooks/
    └── analysis.ipynb        # Exploratory analysis
```

---

## 🖥️ Dashboard Pages

| Page | What it shows |
|------|--------------|
| 🏠 Overview | Problem framing, pipeline diagram, how VeriFlow works |
| 📊 BTP Analytics | KPIs, before/after triage funnel, station rejection rates, hourly trends |
| 🎯 Triage Simulator | Train the model live + simulate any violation scenario |
| 🖼️ CV Detector | Upload image → detect vehicles → classify violations → triage each flag |
| 🗺️ Heatmap | Bengaluru GPS map of approved vs rejected violations |

---

## 🔬 Tech Stack

| Component | Technology |
|-----------|-----------|
| Vehicle detection | YOLOv8n (Ultralytics) |
| Violation classification | Rule-based + lightweight classifier |
| Triage model | Gradient Boosting + Isotonic Calibration |
| ANPR / OCR | EasyOCR + PaddleOCR |
| Dashboard | Streamlit |
| Charts | Plotly |
| Maps | Folium + streamlit-folium |

---

## 📈 Evaluation Metrics

| Metric | Value |
|--------|-------|
| Classification Accuracy | 70.1% |
| Precision | 70.8% |
| Recall | 97.5% |
| F1 Score | 82.0% |
| Auto-Clear Accuracy | 84.5% |
| Queue Reduction Rate | 15.3%+ |

---

## 🔭 Future Roadmap

- **Multi-frame temporal consistency** — cross-check across 3 consecutive frames
- **Texture/segmentation re-check** — fix the documented seatbelt vs. shirt-colour false positive using SAM-based crop segmentation
- **Modular violation plugins** — add BTP's 6 new violation types (7→13) as config changes, not retrains
- **Parking → Congestion impact score** — link illegal parking flags to lane blockage severity
- **Field officer app** — extend to body-worn camera / FTVR device feeds

---

## 👥 Team

Built for **Flipkart Gridlock Hackathon 2.0** · June 2026
In partnership with **Bengaluru Traffic Police (ASTraM)**
