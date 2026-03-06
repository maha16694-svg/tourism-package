
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="maha16694-svg/tourism-package",
    filename="tourism_package_model.joblib"
)

# Load model
model = joblib.load(model_path)

# ------------------------------
# Streamlit UI
# ------------------------------

st.title("Tourism Package Prediction App")

st.write("""
This application predicts whether a customer is likely to purchase the **Wellness Tourism Package**.
Enter the customer details below to get the prediction.
""")

# ------------------------------
# User Inputs
# ------------------------------

Age = st.number_input("Age", 18, 80, 30)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Freelancer", "Small Business", "Large Business"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

NumberOfPersonVisiting = st.number_input(
    "Number Of Persons Visiting",
    1, 10, 2
)

PreferredPropertyStar = st.slider(
    "Preferred Hotel Star Rating",
    1, 5, 3
)

NumberOfTrips = st.number_input(
    "Number Of Trips Per Year",
    0, 20, 2
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    1, 5, 3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe"]
)

NumberOfFollowups = st.number_input(
    "Number Of Followups",
    0, 10, 2
)

DurationOfPitch = st.number_input(
    "Duration Of Pitch (minutes)",
    1, 60, 10
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "VP", "AVP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    10000, 500000, 50000
)

# ------------------------------
# Create DataFrame
# ------------------------------

input_data = pd.DataFrame([{

    "Age": Age,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome

}])

# ------------------------------
# Prediction
# ------------------------------

if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Customer is likely to purchase the Wellness Tourism Package.")
    else:
        st.warning("Customer is unlikely to purchase the package.")
