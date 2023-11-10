import pandas as pd
import streamlit as st
import pickle as pk

pipe = pk.load(open("imported/pipeline.pkl","rb"))
df = pk.load(open("imported/dataframe.pkl","rb"))

# title
st.title("Loan Approval Prediction")
st.write("Welcome to Loan Approval Prediction app. This app allows banks to check whether they need to give loan to peoples or not based on following features.")
st.header("Feature Selection")

# gender 1:Male , 0:Female
gender = st.selectbox("Gender", df["Gender"].unique())

# married 1:Yes , 0:No
married = st.selectbox("Married", df["Married"].unique())

# dependents
dependents = st.number_input("Dependents")

# education 1:Graduate , 0:Not Graduate
education = st.selectbox("Education", df["Education"].unique())

# self employed 1:Yes , 0:No
self_employed = st.selectbox("Self Employed", df["Self_Employed"].unique())

# applicant income
applicant_income = st.number_input("Applicant Income")

# co-applicant income
co_applicant_income = st.number_input("Co-Applicant Income")

# loan amount
loan_amount = st.number_input("Loan Amount")

# loan amount term
loan_amount_term = st.number_input("Loan Amount Term")

# credit History
credit_history = st.selectbox("Credit History", ["Yes","No"])

# property area 0:Rural , 1:SemiUrban , 2:Urban
property_area = st.selectbox("Property Area", df["Property_Area"].unique())

# button 
if st.button("Check Loan Eligibility"):

    if credit_history=="Yes":
        credit_history=1
    else:
        credit_history=0

    user_df = pd.DataFrame({"Gender":[gender],
                            "Married":[married],
                            "Dependents":[int(dependents)],
                            "Education":[education],
                            "Self_Employed":[self_employed],
                            "ApplicantIncome":[int(applicant_income)],
                            "CoapplicantIncome":[int(co_applicant_income)],
                            "LoanAmount":[int(loan_amount)],
                            "Loan_Amount_Term":[int(loan_amount_term)],
                            "Credit_History":[int(credit_history)],
                            "Property_Area":[property_area]})

    ans = pipe.predict(user_df)[0]

    if ans==0:
        ans="Not Eligible"
    else:
        ans="Eligible"

    st.header(f"{ans} for Loan.")


# Note or Disclaimer
st.header("Disclaimer")
st.write("This app provides estimates guesses based on the dataset and machine learning model used. The actual status may vary.")

# Contact Information or About the Developer (Optional)
st.header("About the Developer")
st.write("This app was developed by Abdullah_Khan_Kakar. You can contact him at abdullahkhan4465917@gmail.com for any questions or feedback.")
