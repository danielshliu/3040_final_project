import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Telecom Customer Churn Dataset Dashboard")

#load the dataset
def load_data():
    old_df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv.xls")
    df = pd.read_csv("preprocessed_telco_churn.csv")
    #st.dataframe(df)
    return df, old_df
old_df, df = load_data()


#Section: Overview of the date
st.header("Overview")
#total number of customers
total = df["gender"].value_counts().sum()
st.write(f"Total customers: {total}")


#find the Churn rate 
churn = old_df["Churn"].value_counts()
churn.index = ["No", "Yes"]
#pie chart for churn rate
fig, ax = plt.subplots(figsize=(2, 3))
ax.pie(churn.values, labels=churn.index, autopct="%1.1f%%")
ax.set_title("Customer Churn rate")
st.pyplot(fig)


#bar chart contract type distribution
st.write("Contract Type Distribution")
contract = df["Contract"].value_counts()
contract_df = pd.DataFrame({
    "Contract Type": contract.index,
    "Count": contract.values
})
contract_df = contract_df.set_index("Contract Type")
st.bar_chart(contract_df)


#histogram for Tenure distribution 
st.write("Tenure Distribution (in months)")
bins = [0, 12, 24, 36, 48, 60, 72]
labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
df["trange"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)

tenure_counts = df["trange"].value_counts().sort_index()
tenure_df = pd.DataFrame({
    "Tenure": tenure_counts.index,
    "Count": tenure_counts.values
})
tenure_df = tenure_df.set_index("Tenure")
st.bar_chart(tenure_df)


#Section Results: evaluation of models
#For decision tree model
st.header("Results")
st.subheader("Using Decision Tree model:")
a, b = st.columns(2)
c, d = st.columns(2)
#accuracy 
a.metric("Accuracy","73%")
#presicion
b.metric("Presicion", "80%")
#recall
c.metric("Recall", "73%")
#confusion matrix
cm = np.array([[733, 302], [72, 302]])
cm_df = pd.DataFrame(cm, columns=["Predicted 0 (No Churn)", "Predicted 1 (Churn)"], index=["Actual 0 (No Churn)", "Actual 1 (Churn)"])
d.write("Confusion Matrix:")
d.dataframe(cm_df)

#For logistic regression model
st.subheader("Using Logistic Regression model:")
e, f = st.columns(2)
g, h = st.columns(2)
#accuracy 
e.metric("Accuracy","80%")
#presicion
f.metric("Presicion", "60%")
#recall
g.metric("Recall", "50%")
#confusion matrix
cm2 = np.array([[923, 112], [192, 182]])
cm2_df = pd.DataFrame(cm2, columns=["Predicted 0 (No Churn)", "Predicted 1 (Churn)"], index=["Actual 0 (No Churn)", "Actual 1 (Churn)"])
h.write("Confusion Matrix:")
h.dataframe(cm2_df)


#Secton importatn facture (maybe bar chart)
importances = pd.read_csv("dt_importances.csv")
importances = importances[importances["Importance"]>0]
fig1, ax1 = plt.subplots()
ax1.barh(importances["Feature"], importances["Importance"])
ax1.set_xlabel("Importance")
ax1.set_ylabel("Feature")
ax1.set_title("Decision Tree Feature Importances")
st.pyplot(fig1)
