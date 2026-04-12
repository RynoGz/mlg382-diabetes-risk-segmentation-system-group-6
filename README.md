# 🩺 Diabetes Risk Segmentation & Decision Support System

## 📖 Project Overview

This project was developed as part of a data science initiative at **BC Analytics**, a health-tech startup focused on improving patient outcomes and reducing long-term healthcare costs.

The goal of this system is to support early detection and management of diabetes risk by combining machine learning, data analysis, and an interactive dashboard.

---

## ❗ Problem Statement

Healthcare providers often:

* React too late to diabetes warning signs
* Lack clear insights into lifestyle risk factors
* Do not have data-driven tools during consultations

This project aims to address these challenges using predictive modeling and patient segmentation.

---

## 🎯 Objectives

* Predict diabetes risk level (`diabetes_stage`)
* Segment patients into meaningful risk categories
* Identify key lifestyle factors influencing diabetes risk
* Group patients based on lifestyle and health characteristics
* Provide actionable insights through a web dashboard

---

## 🧠 Methodology (CRISP-DM)

### 1. Data Understanding

* Explored dataset features and diabetes stages
* Researched different diabetes types in the dataset

### 2. Data Preparation

* Data cleaning and preprocessing
* Handling missing values
* Feature engineering

### 3. Modeling

#### 🔹 Classification Models

* Decision Tree
* Random Forest
* XGBoost

#### 🔹 Clustering

* K-Means Clustering (k = 3)

### 4. Evaluation

* Model performance metrics (accuracy, precision, recall, F1-score)
* Cluster interpretation and validation

### 5. Explainability

* SHAP (Shapley Additive Explanations)
* Identification of key lifestyle risk drivers

### 6. Deployment

* Interactive dashboard built using Dash
* Model integration for real-time predictions

---

## 🤖 Models Used

* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier
* K-Means Clustering

---

## 📊 Key Insights

* Identification of major lifestyle factors influencing diabetes risk
* Clear grouping of patients into meaningful segments
* Improved interpretability of model predictions using SHAP

---

## 🌐 Web Application

👉 [Insert Render Link Here]

---

## 🎥 Video Demonstration

👉 [Insert Video / OneDrive Link Here]

---

## 📁 Project Structure

data/        → Raw and processed datasets
notebooks/   → Jupyter notebooks for each phase
src/         → Python scripts (clean implementation)
models/      → Saved machine learning models
app/         → Dash web application
reports/     → Technical report
docs/        → Video link and supporting files

---

## 👥 Team Responsibilities

### 🔹 Data Engineer

* Data preprocessing and cleaning
* Feature engineering
* Work in:

  * data/
  * notebooks/01_data_understanding.ipynb
  * notebooks/02_preprocessing.ipynb
  * src/preprocessing.py

---

### 🔹 Lead Classifier

* Build and evaluate classification models
* Hyperparameter tuning
* Work in:

  * notebooks/03_classification.ipynb
  * src/train_models.py
  * models/classification_model.pkl

---

### 🔹 Segmentation Lead

* Perform patient segmentation using K-Means (k = 3)
* Interpret cluster results
* Work in:

  * notebooks/04_clustering.ipynb
  * src/clustering.py
  * models/kmeans_model.pkl

---

### 🔹 XAI Specialist

* Perform SHAP analysis
* Interpret model predictions
* Work in:

  * notebooks/05_xai.ipynb
  * src/explainability.py

---

### 🔹 Web Developer

* Develop Dash web application
* Integrate trained models
* Deploy application
* Work in:

  * app/app.py
  * app/requirements.txt

---

### 🔹 Project Lead

* Repository structure and management
* Documentation and report compilation
* Ensure consistency and quality
* Final submission and links

---

## ⚙️ How to Run the Project

### 1. Clone the repository

git clone https://github.com/your-repo-link.git
cd diabetes-risk-segmentation-system

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the application

python app/app.py

---

## 📌 Version Control

* GitHub is used for collaboration and task management
* Each team member contributes with meaningful commits
* Repository follows a structured and modular design

---

## ✅ Deliverables

* Technical Report (CRISP-DM based)
* GitHub Repository
* Interactive Dash Web Application
* Video Demonstration

---

## 🚀 Future Improvements

* Integration with real healthcare systems
* Real-time data updates
* Enhanced model performance and tuning
* Expanded dataset for better generalization
