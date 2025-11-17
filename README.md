# 📘 Churn Prediction Using Deep Learning  
### A-Deep-Learning-Project---Customer-Churn-Prediction

<p align="center">
  <img width="800" src="https://github.com/user-attachments/assets/d4f61d54-3ed0-43c9-80d2-7524976aeca0" />
  <br/>
  <img width="800" src="https://github.com/user-attachments/assets/d9f70130-7b57-46b7-b231-0c9fa92214e5" />
</p>

---

## 🚀 Project Overview

This project is a **full-stack machine learning system** that builds and deploys a deep learning model to predict customer churn.

It includes:

- A Python script for **data processing and ANN model training**
- A **Flask backend** to serve the trained model as a REST API
- A **HTML/CSS/JavaScript frontend** (with Tailwind CSS) for real-time user predictions

---

## ❓ What Is Churn Prediction?

**Customer churn** refers to when customers stop using a service or leave a company (e.g., cancel a subscription).

**Churn prediction** uses machine learning to identify customers likely to leave.  
Companies use these predictions to:

- Offer discounts  
- Improve customer experience  
- Prevent revenue loss  

This makes churn prediction a crucial tool for subscription-based businesses.

---

## 🛠️ Technology Stack

| Component | Technology Used |
|----------|------------------|
| **Model** | Artificial Neural Network (TensorFlow / Keras) |
| **Backend** | Flask |
| **Frontend** | HTML, JavaScript, Tailwind CSS |
| **Data Processing** | Pandas, Scikit-Learn |
| **Deployment Files** | `churn_model.keras`, `preprocessor.joblib` |

---

## 🧠 The Model & Dataset

### 🔹 Machine Learning Model

The project uses a **Deep Learning ANN (Artificial Neural Network)** built with:

- TensorFlow  
- Keras Sequential API  
- Dense Layers  
- Dropout Regularization  

The model effectively captures complex patterns in customer behavior.

---

### 🔹 Dataset Used

This project uses the popular **Telco Customer Churn Dataset**  
(Source: IBM | Available on Kaggle)

**Dataset Features Include:**

- **Demographics:** gender, SeniorCitizen, Partner, Dependents  
- **Account Details:** tenure, Contract, MonthlyCharges, PaymentMethod  
- **Subscribed Services:** InternetService, PhoneService, OnlineSecurity, TechSupport, etc.  

📌 **Total Records:** 7043  
📌 **Features:** 21  

---

## 📦 Flask API (Backend)

The Flask backend is responsible for serving the deep learning model.

### ✔️ Key Functions:

- Load `churn_model.keras` (trained ANN)
- Load `preprocessor.joblib` (Scikit-learn pipeline)
- Provide a **REST API endpoint**:  
