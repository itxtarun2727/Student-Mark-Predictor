# import libraries
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# page title
st.set_page_config(page_title="Student Mark Predictor", layout="centered")

st.title("🎓 Student Marks Prediction App")
st.write("Predict student marks based on study hours using Machine Learning")

# load dataset
df = pd.read_csv("student_info.csv")

# handle missing values
df = df.fillna(df.mean())

# split data
X = df.drop("student_marks", axis=1)
Y = df.drop("study_hours", axis=1)

# train model
lr = LinearRegression()
lr.fit(X, Y)

# USER INPUT (THIS REPLACES input())
study_hours = st.number_input(
    "Enter study hours (between 3 and 12)",
    min_value=0.0,
    max_value=24.0,
    step=0.1
)

# prediction button
if st.button("Predict Marks"):
    if study_hours >= 3 and study_hours <= 12:
        result = lr.predict([[study_hours]])[0][0]
        st.success(f"📊 Expected Marks: {round(result, 2)}")
    else:
        st.error("❌ Please enter study hours between 3 and 12")
