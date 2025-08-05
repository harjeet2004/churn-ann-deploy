# Customer Churn Prediction using ANN

This is a simple machine learning project that predicts whether a bank customer is likely to churn using an Artificial Neural Network (ANN) and a user-friendly Streamlit interface.

## 🔗 Live App

Try the app here:  
https://churn-ann-deploy-jt2ktjxas5rjwhg2rzwwvm.streamlit.app/

## 📝 Project Description

This app takes customer details like Age, Gender, Geography, Credit Score, Account Balance, Salary, etc., and predicts whether the customer will churn or not. The backend is powered by a deep learning model trained using TensorFlow and deployed via Streamlit Cloud.

## 🛠 Technologies Used

- Python 3.10
- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Pickle
- Git & GitHub
- Streamlit Cloud

## 📁 Project Structure

```
.
├── app.py                     # Main Streamlit application
├── model.h5                   # Trained ANN model
├── scaler.pkl                 # StandardScaler saved model
├── label_encoder_gender.pkl   # LabelEncoder for Gender
├── onehot_encoder_geo.pkl     # OneHotEncoder for Geography
├── requirements.txt           # Python dependencies
├── runtime.txt                # Python version (for Streamlit Cloud)
├── .gitignore                 # Files to ignore during Git push
└── README.md                  # This file
```

## 💻 How to Run Locally

Follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create virtual environment and activate it

```bash
conda create -n churn-env python=3.10
conda activate churn-env
```

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Then open your browser and visit:  
http://localhost:8501

## 🚀 How to Deploy on Streamlit Cloud

1. Push your project to a GitHub repository.
2. Go to https://streamlit.io/cloud and sign in.
3. Click on “New app” and select your repo.
4. Set the main file as `app.py`.
5. Ensure `requirements.txt` and `runtime.txt` are present.
6. Click “Deploy”.

## 🧠 What I Learned

- How to build and train an ANN model
- How to save and reuse encoders and scalers using pickle
- How to use Streamlit to build web apps
- How to deploy machine learning projects online

## 🙋‍♂️ Author

Built by a college student learning machine learning and web app deployment.


This project is licensed under the MIT License.
