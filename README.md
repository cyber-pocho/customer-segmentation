# Customer Segmentation Analysis Using RFM and Unsupervised Learning

## 🧭 Project Overview

This repository presents a structured approach to customer segmentation through **Recency, Frequency, and Monetary (RFM)** analysis and unsupervised machine learning techniques. The implementation showcases skills in object-oriented programming, statistical data processing, and business intelligence development. It serves as a robust portfolio project in applied data science.

## 📊 Business Context

Customer segmentation is a fundamental strategy in customer relationship management. It allows organizations to:

- **Personalize Marketing**: Design and deliver targeted promotional content
- **Optimize Resource Allocation**: Prioritize high-value customer cohorts
- **Improve Customer Retention**: Detect and re-engage at-risk customers
- **Increase Revenue**: Tailor strategies that maximize value across customer segments

## 🛠️ Technical Stack

- **Python**: Primary programming environment
- **Pandas**: Data wrangling and preprocessing
- **NumPy**: Efficient numerical computation
- **Scikit-learn**: Clustering algorithms and evaluation metrics
- **Matplotlib & Seaborn**: Visual analytics and exploratory data analysis

## ⚙️ Core Features

- **Object-Oriented Architecture**: Encapsulated in a modular `CustomerSegmentationAnalyzer` class
- **End-to-End Automation**: From raw data ingestion to insight generation
- **Multiple Clustering Techniques**: Implementation of both KMeans and DBSCAN
- **Model Selection Tools**: Elbow method and silhouette scoring for hyperparameter tuning
- **Insightful Visualizations**: Heatmaps, histograms, cluster profiles, and correlation matrices
- **Data-Driven Recommendations**: Business actions informed by segmentation outputs

## 📈 Methodological Framework

### 1. Data Preprocessing

- Removal of return transactions and invalid invoice entries
- Elimination of duplicates and imputation of missing values
- Construction of derived variables, such as `TotalAmount`

### 2. RFM Feature Engineering

- **Recency**: Days elapsed since the last transaction (lower values preferred)
- **Frequency**: Count of unique transactions per customer
- **Monetary**: Aggregate customer spending
- Features normalized via MinMax scaling to standardize input space for clustering

### 3. Clustering Analysis

- Application of **KMeans** clustering (k optimized using Elbow and Silhouette methods)
- Exploratory use of **DBSCAN** for density-based segmentation
- Final model selected with **k = 4**, yielding a silhouette score of **[X.XXX]**

### 4. Segment Profiling

Customer segments were identified and labeled based on statistical properties:

- 💎 **VIP Champions**: High frequency, high spending, recent purchasers
- 🌟 **Loyal Customers**: Consistent purchasing patterns with moderate-to-high value
- 😴 **Lost Customers**: No recent activity; historically active
- 🆕 **New Customers**: Recent first-time purchasers; growth opportunity

## 🔍 Segment Insights

### Strategic Recommendations

| Segment         | Recommended Strategy                                            |
|-----------------|------------------------------------------------------------------|
| VIP Champions   | Design exclusive loyalty tiers with premium perks               |
| Loyal Customers | Introduce a points-based reward system with referral incentives |
| Lost Customers  | Launch win-back campaigns with targeted discounts               |
| New Customers   | Offer onboarding journeys with bundled or cross-sell offers     |

## 📁 Repository Structure

├── data/
│   └── raw/                  # Raw transactional data files
├── src/
│   └── segmentation.py    
│   └── analysis.ipynb   # Core class and analysis logic
├── visuals/
│   └── *.png                 # Cluster visualizations and plots
├── README.md
└── requirements.txt
