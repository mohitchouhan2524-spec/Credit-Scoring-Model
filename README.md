# 💳 Credit Scoring Model

**Predict your credit risk with machine learning.**

A Credit Scoring Model that analyzes financial information and predicts the likelihood of credit risk using multiple machine learning algorithms. The application provides an intuitive web interface built with **Streamlit** and securely stores prediction history using **Supabase**.

---

## 📌 Overview

This project demonstrates an end-to-end machine learning pipeline for credit risk assessment, covering everything from data preprocessing to deployment.

The workflow includes:

* 📊 Data Analysis
* 🛠️ Feature Engineering
* 🤖 Model Training
* 📈 Credit Risk Prediction
* 🌐 Interactive Web Application
* ☁️ Cloud Database Integration

---

## 🚀 Features

* Predicts customer credit risk in real time.
* Clean and interactive Streamlit interface.
* Secure user authentication using Supabase.
* Stores prediction history for authenticated users.
* Multiple machine learning models for comparison.
* Fast and lightweight deployment.

---

## 🧠 Machine Learning Pipeline

### Data Analysis

* Exploratory Data Analysis (EDA)
* Missing value handling
* Outlier inspection
* Feature distribution analysis

### Feature Engineering

* Data preprocessing
* Categorical encoding
* Feature scaling
* Feature selection

### Model Training

The following machine learning models were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

The best-performing model is used to generate credit risk predictions.

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn

### Frontend

* Streamlit

### Backend & Database

* Supabase

---

## 📂 Project Structure

```text
Credit-Scoring-Model/
│
├── app/
├── data/
├── models/
├── notebooks/
├── src/
├── .streamlit/
├── requirements.txt
├── README.md
└── app.py
```

---

## ⚙️ How It Works

1. User signs in to the application.
2. Financial information is entered through the web interface.
3. The data is preprocessed using the same feature engineering pipeline used during training.
4. The trained machine learning model predicts the credit risk.
5. Prediction results are displayed instantly.
6. Prediction history is securely stored in Supabase.

## 🌍 Live Demo

**Website:**
*https://credit-scoring-model-2k26.streamlit.app*

---

## 📦 GitHub Repository

**GitHub:**
*https://github.com/mohitchouhan2524-spec/Credit-Scoring-Model.git*

## ⭐ If you found this project helpful, consider giving it a star!
