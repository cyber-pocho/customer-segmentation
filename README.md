# Customer Segmentation with RFM and Machine Learning

## 🔍 Overview

This project applies **RFM (Recency, Frequency, Monetary)** analysis and unsupervised learning to identify customer segments and generate actionable business insights. Built with a modular, object-oriented structure, it automates the entire process — from raw data cleaning to strategy recommendations.

Ideal as a real-world portfolio project for business intelligence, customer analytics, and CRM consulting.

---

## 💼 Business Impact

Customer segmentation helps businesses:

- 🎯 **Target marketing campaigns** to specific customer types
- 💰 **Boost retention and revenue** through personalized strategies
- 🧠 **Understand behavioral patterns** across the customer lifecycle
- 📉 **Reduce churn** by identifying at-risk segments

---

## 🛠️ Tools & Technologies

| Tool          | Purpose                           |
|---------------|------------------------------------|
| Python        | Core language                     |
| Pandas, NumPy | Data wrangling and transformation |
| Scikit-learn  | Clustering models (KMeans, DBSCAN)|
| Matplotlib, Seaborn | Visual analytics             |

---

## 🔧 Key Features

- Encapsulated in a reusable `CustomerSegmentationAnalyzer` class
- Implements **KMeans** and **DBSCAN** clustering
- Model tuning with Elbow method & Silhouette scores
- Visual summaries: heatmaps, histograms, correlation plots
- Business-oriented **segment labeling** and **recommendations**

---

## 🧪 Methodology

### 1. Data Preprocessing
- Removal of returns, invalid invoices, duplicates
- Imputation of missing values
- Feature creation (e.g., `TotalAmount`)

### 2. RFM Calculation
- **Recency** = days since last purchase
- **Frequency** = number of transactions
- **Monetary** = total spend
- Features normalized via MinMax scaling

### 3. Clustering
- KMeans clustering with optimal `k` selected via:
  - Elbow method
  - Silhouette score (final score: **X.XXX**)
- DBSCAN tested for non-linear clusters

### 4. Segment Profiling

| Segment         | Description                                           |
|-----------------|-------------------------------------------------------|
| 💎 VIP Champions | High frequency, high value, recent purchases         |
| 🌟 Loyal Customers | Repeat buyers, consistent spend                     |
| 😴 Lost Customers  | No recent activity, historically active             |
| 🆕 New Customers   | First-time buyers with potential for growth         |

---

## 💡 Business Recommendations

| Segment         | Strategy                                              |
|-----------------|-------------------------------------------------------|
| VIP Champions   | Loyalty tiers, premium perks                          |
| Loyal Customers | Referral incentives, exclusive deals                  |
| Lost Customers  | Win-back campaigns with discounts                     |
| New Customers   | Onboarding flows, bundles, cross-sells                |

---

## 📂 Project Structure
├── data/
│ └── raw/ # Raw transactional files
├── src/
│ ├── segmentation.py # Core class
│ └── analysis.ipynb # Notebook walkthrough
├── visuals/ # Cluster plots
├── requirements.txt
└── README.md

## ▶️ How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Open and run the Jupyter notebook:
    src/analysis.ipynb

## 📬 Contact

📌 **Author**: Julian Alfonso y Gomez  
📧 **Email**: juliandavid.alfonso.gomez@gmail.com  
🕒 **Last Updated**: July 28, 2025

> This project demonstrates advanced customer analytics and unsupervised learning — with real business application.
