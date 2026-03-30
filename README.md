# Customer Segmentation & Retention Analyzer

Machine learning web application predicting telecom customer churn (96.3% accuracy) and segmenting customers into three value groups.

## Key Results
- Dataset: 667 telecom customers
- Model Accuracy: 96.3% 
- Predicted Churn: 92 customers (13.8%)
- Segments: 263 High-value (39%), 211 Low (32%), 193 Medium (29%)

## Features
- Churn prediction using Random Forest
- Customer segmentation with KMeans clustering
- Responsive web dashboard with Bootstrap
- High-risk customer identification

## Tech Stack
Python
Scikit-learn
Flask
Pandas
Bootstrap 5
Matplotlib

## Quick Start
```bash
pip install -r requirements.txt
python model.py    # Train model (96.3% accuracy)
python app.py      # http://127.0.0.1:5000
```


## Model Performance# customer-segmentation-retention-analyzer
Test Accuracy: 96.3%
Churn Precision: 92% | Recall: 73% | F1: 81%

## Business Impact
- Identifies 92 high-risk customers for retention
- Prioritizes 263 high-value customers (39% of base)
- Production-ready deployment pipeline

  ## File Structure
  customer-analyzer/
  │── app.py
  │── model.py
  │── dataset.csv
  │── templates/
  │── static/
  │── README.md

  ## Next Steps
- Live deployment (Render/Heroku)
- Model explainability (SHAP)
- Retention recommendations

Built by Aarti Singh | Panaji, Goa

