# 🧠 Customer Churn & Salary Prediction using ANN

This repository contains two AI-powered web applications built with Streamlit:

1. **Customer Churn Classification**  
2. **Salary Regression Prediction**

Both use Artificial Neural Networks (ANNs) built with TensorFlow and trained on structured customer data. The models are deployed on Streamlit Cloud for easy access.

---

## 🔗 Live Apps

- **🔄 Churn Classification App**  
  👉 [Churn-Classification-app](https://churn-ann-deploy-jt2ktjxas5rjwhg2rzwwvm.streamlit.app/)

- **💰 Salary Regression App**  
  👉 [Salary-Prediction-app](https://churn-ann-deploy-zvstvywfxjgr2zxussg3pt.streamlit.app/)

---

## 📝 Project Descriptions

### 🔄 1. Churn Classification

Predicts whether a bank customer is likely to churn based on input parameters like:
- Credit Score
- Gender
- Age
- Balance
- Salary
- Geography
- Tenure
- Number of Products
- Credit Card Ownership
- Active Membership

### 💰 2. Salary Regression

Predicts the estimated salary of a customer using similar features, but trained on a regression-targeted ANN. Instead of churn, the model outputs a continuous value — predicted salary.

---

## 🛠 Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras**
- **Streamlit**
- **Pandas & NumPy**
- **Scikit-learn**
- **Pickle**
- **Git & GitHub**
- **Streamlit Cloud**

---

## 📁 Project Structure

```
.
├── app.py                    # Churn classification Streamlit app
├── app_reg.py                # Salary regression Streamlit app
├── model.h5                  # Classification model (ANN)
├── regression_model.h5       # Regression model (ANN)
├── scalerclass.pkl           # Scaler for classification inputs
├── scalereg.pkl              # Scaler for regression inputs
├── label_encoder_gender.pkl  # LabelEncoder for gender
├── onehot_encoder_geo.pkl    # OneHotEncoder for geography
├── Churn_Modelling.csv       # Original dataset
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore
└── notebooks/
    ├── Churnclassification.ipynb   # Classification model training
    ├── salaryregression.ipynb      # Regression model training
    └── prediction.ipynb            # Common prediction utilities
```

---

## 💻 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/churn-ann-deploy.git
cd churn-ann-deploy
```

### 2. Create & Activate a Virtual Environment

```bash
conda create -n ann-env python=3.10
conda activate ann-env
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run Either App Locally

```bash
# For churn classification
streamlit run app.py

# For salary regression
streamlit run app_reg.py
```

Visit:  
[http://localhost:8501](http://localhost:8501)

---

## 🚀 How to Deploy on Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click “New app” and choose your repo
4. Set the correct main file (`app.py` or `app_reg.py`)
5. Add `requirements.txt` and optionally `runtime.txt`
6. Click **Deploy**

---

## 🧠 What I Learned

- Building and training classification & regression ANN models
- Handling encoders and scalers using `pickle`
- Best practices for organizing ML projects
- Creating clean, user-friendly UIs using Streamlit
- Fixing deployment issues related to model deserialization
- Publishing multiple apps on Streamlit Cloud from a single repo

---

## 🙋‍♂️ Author

Built with ❤️ by a college student exploring AI, data science, and real-world ML deployment.

---

## 📜 License

This project is licensed under the **MIT License** — feel free to fork, extend, and contribute!
