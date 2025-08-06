## End=to-end Streamlit web app

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model=load_model('regression_model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    gen_lab=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot=pickle.load(file)

with open('scalerreg.pkl','rb') as file:
    scaler=pickle.load(file)

## streamlit app

st.title('Customer Salary Prediction')

##############################################################
### 📦 STREAMLIT USER INPUT INTERFACE FOR PREDICTION MODEL ###
##############################################################

# ✅ Assume the following are preloaded using pickle:
# - `onehot` → trained OneHotEncoder for Geography
# - `gen_lab` → trained LabelEncoder for Gender
# - `scaler` → trained StandardScaler for numerical scaling
# - `model` → trained classification model (e.g., ANN, RandomForest)

# 🔷 1. GEOGRAPHY INPUT
# Show dropdown for 'Geography' using the trained OneHotEncoder

# onehot.categories_ returns a list of arrays (1 per column encoded)
# Since we encoded only one column ('Geography'), we use categories_[0] to access it
# e.g., onehot.categories_ → [array(['France', 'Germany', 'Spain'], dtype=object)]
# so, onehot.categories_[0] → ['France', 'Germany', 'Spain']

geography = st.selectbox(
    'Geography',             # 🏷️ Label shown on UI
    onehot.categories_[0]    # ✅ Known values learned during training
)

# ✅ GENERAL SYNTAX:
# encoder.categories_[column_index] → returns list of categories for a column (used in OneHotEncoder)

# 🧠 WHY WE USE THIS:
# To ensure only known categories (used during model training) are shown to the user.
# If we use an unseen value (e.g., "Italy"), onehot.transform() will raise an error.
# This guarantees compatibility with the encoder.

# ❗ DO NOT use encoder directly in DataFrame like: pd.DataFrame(onehot)
# Because:
# ➤ onehot is the **machine** (stamp maker)
# ➤ You need to use: `geo_encoded = onehot.transform([[geography]])` → This is the **stamped output**
# ➤ Then wrap in a DataFrame with column names from: onehot.get_feature_names_out(['Geography'])

# ✅ CORRECT WAY (for later step):
# geo_encoded = onehot.transform([[geography]])
# geo_df = pd.DataFrame(geo_encoded, columns=onehot.get_feature_names_out(['Geography']))


# 🔷 2. GENDER INPUT
# Use LabelEncoder to handle gender ('Male', 'Female')
# gen_lab.classes_ returns array of labels it was trained on
# e.g., gen_lab.classes_ → ['Female', 'Male']

gender = st.selectbox(
    'Gender',
    gen_lab.classes_         # ✅ Consistent labels from training
)

# ✅ GENERAL SYNTAX:
# label_encoder.classes_ → returns array of known class labels (used in LabelEncoder)

# 🧠 WHY WE USE THIS:
# Again, we want to restrict inputs to known values
# This avoids mismatch errors during: gen_lab.transform([gender])

# ✅ CORRECT WAY (for later step):
# gender_encoded = gen_lab.transform([gender])  # will return [0] or [1] depending on label


# 🔷 3. AGE INPUT (Slider)
age = st.slider(
    'Age',
    min_value=18,
    max_value=92
)

# ✅ GENERAL SYNTAX:
# st.slider('Label', min_value, max_value)

# 🎯 Use sliders for bounded integer inputs, like age


# 🔷 4. BALANCE INPUT (Free number input)
balance = st.number_input('Balance')

# ✅ GENERAL SYNTAX:
# st.number_input('Label')


# 🔷 5. CREDIT SCORE
credit_score = st.number_input('Credit Score')


# 🔷 6. ESTIMATED SALARY
exited = st.selectbox(
    'Exited',
    [0,1]       
)

# 🔷 7. TENURE (Slider: 0–10 years)
tenure = st.slider('Tenure', 0, 10)


# 🔷 8. NUMBER OF PRODUCTS USED
num_of_products = st.slider('Number of Products', 1, 4)


# 🔷 9. CREDIT CARD STATUS (0 = No, 1 = Yes)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])

# ✅ GENERAL SYNTAX:
# st.selectbox('Label', [options_list])

# 🎯 Use selectbox for categorical binary values


# 🔷 🔟 ACTIVE MEMBER STATUS (0 = No, 1 = Yes)
is_active_member = st.selectbox('Is Active Member', [0, 1])


##############################################################
### 🔁 SUMMARY: ENCODER SYNTAX REFERENCE FOR TRANSFORMATION ###
##############################################################

# 📘 OneHotEncoder
# ----------------
# ➤ onehot.categories_ → list of arrays, each containing unique categories per column
# ➤ onehot.categories_[0] → gives list of geography categories used during fit()
# ➤ onehot.transform([[input_value]]) → returns encoded array (e.g., [1, 0, 0])
# ➤ onehot.get_feature_names_out(['Geography']) → returns encoded column names
#
# 🧠 Why use categories_[0]?
# To ensure input matches known values — avoids transform errors.
# We need this when taking user input from dropdown so that input is "valid".
#
# 📘 LabelEncoder
# ----------------
# ➤ gen_lab.classes_ → list of class labels (e.g., ['Female', 'Male'])
# ➤ gen_lab.transform(['Male']) → returns encoded label (e.g., [1])
# ➤ gen_lab.inverse_transform([1]) → returns label (e.g., 'Male')
#
# 🧠 Why use classes_?
# To ensure user selects from valid labels — no risk of transform failing.


# 🧠 PREPARING INPUT DATA FOR MODEL PREDICTION IN STREAMLIT

# In this Streamlit app, we're collecting user input **one by one** through sliders, dropdowns, etc.

# 👉 These inputs are single values (scalars), like:
#     age = 40
#     credit_score = 600
#     balance = 50000
# But the ML model expects data in the form of a **table** — even if it's just one row.
# That means we need to structure the input as a 1-row, multi-column DataFrame.

# ✅ WHY WE USE [ ] AROUND EACH VALUE:
# Wrapping each value in [ ] turns it into a list → which pandas treats as one row.
# Example: [age] → [40] becomes one row with column 'Age'.

# ✅ WHY WE USE gen_lab.transform([gender])[0]:
# LabelEncoder needs input in list form → so we do [gender]
# It returns something like [1] → so we use [0] to extract just the number
# Then we wrap THAT number again in [ ] to make it fit the DataFrame column.

# -----------------------------------------------
# 🔁 COMPARISON: WHY WE DIDN'T USE [ ] EARLIER?
# -----------------------------------------------
# In earlier notebook-based predictions, we often used:
#     input = [[600, 1, 40, 3, 50000, 2, 1, 1, 60000]]
# That works because:
#     - We manually entered all values in correct order
#     - We already knew the structure
#     - No need for column names if just testing with scaler/model directly

# BUT in Streamlit:
#     - Inputs come from UI fields (not hardcoded)
#     - You build the input row-by-row, value-by-value
#     - So a column-based DataFrame is cleaner, safer, and avoids ordering mistakes

# ✅ Final result: A 1-row DataFrame named `input_data`, ready to merge with encoded geography
# Then we can apply scaling and prediction

input_data = pd.DataFrame({
    'CreditScore': [credit_score],                                 # Turn scalar into 1-row column
    'Gender': [gen_lab.transform([gender])[0]],                    # Encode gender & wrap result in list
    'Age': [age],                                                  # Wrap each value in [ ] to form table row
    'Tenure': [tenure],                                            # Same logic
    'Balance': [balance],                                          # Same logic
    'NumOfProducts': [num_of_products],                            # Same logic
    'HasCrCard': [has_cr_card],                                    # 0 or 1, wrapped in [ ]
    'IsActiveMember': [is_active_member],                          # 0 or 1, wrapped in [ ]
    'Exited': [exited]                          # Final column, wrapped in [ ]
})

# 📌 Now `input_data` is a proper table with shape (1, 9)
# You can combine it with one-hot encoded geography, scale it using StandardScaler,
# and feed it into model.predict() to get the output.

# ---------------------------------------------
# 🧠 STEP 1: ONE-HOT ENCODE THE SELECTED GEOGRAPHY
# ---------------------------------------------

# The user selected a geography (like 'France') via Streamlit
# Our trained model does not understand raw text labels — it was trained using one-hot encoded values.
# So we use the trained OneHotEncoder to convert 'France' into an array like: [[1.0, 0.0, 0.0]]

# ✅ Why double brackets [[geography]]?
# Because OneHotEncoder expects 2D input (rows × columns), 
# even for a single value. So [['France']] works, but ['France'] would raise an error.

geo_enc = onehot.transform([[geography]])


# ---------------------------------------------
# 🧠 STEP 2: CONVERT ENCODED OUTPUT TO A DATAFRAME
# ---------------------------------------------

# The result from encoder is a NumPy array like: [[1.0, 0.0, 0.0]]
# But to combine it with our main input data, we need a pandas DataFrame with proper column names.

# ✅ get_feature_names_out(['Geography']) gives us:
# ['Geography_France', 'Geography_Germany', 'Geography_Spain']
# This helps keep the column names clear and consistent with training.

geo_df = pd.DataFrame(
    geo_enc,
    columns=onehot.get_feature_names_out(['Geography'])
)


# ---------------------------------------------
# 🧠 STEP 3: CONCAT THE ENCODED GEOGRAPHY WITH THE MAIN INPUT DATA
# ---------------------------------------------

# Now we add the one-hot encoded geography columns to our main `input_data` DataFrame.

# ✅ Why not drop "Geography" before adding geo_df?
# Because our `input_data` never had a 'Geography' column to begin with!
# We had collected it separately and encoded it directly — so there's nothing to remove.

# ⚠️ If you tried to do:
# input_data.drop("Geography", axis=1)
# You'd get: KeyError: "['Geography'] not found in axis"

# ✅ Why reset_index(drop=True)?
# This ensures both DataFrames (input_data and geo_df) have matching row indices (0 in this case),
# so that the merge works cleanly — otherwise, pandas may misalign rows or insert NaNs.

# ✅ axis=1 means we are concatenating column-wise (left-to-right), not row-wise (top-to-bottom)

input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_df],  # combine original inputs + encoded geography
    axis=1
)

# ✅ Final Result: input_data now includes all model-ready features:
# - Numeric fields like CreditScore, Age, Salary...
# - Encoded 'Gender'
# - One-hot encoded 'Geography'
# Now it's ready to be scaled and passed into model.predict()


input_data_scaled=scaler.transform(input_data)

prediction=model.predict(input_data_scaled)

pred_salary=prediction[0][0] ## here, no prediction probability as here, continuous value os predicted (regression), no classification

st.write(f"Estimated salary of the employee is: {pred_salary}")