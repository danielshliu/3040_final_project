#------------------------------------------------- Retrieving the data -------------------------------------------------
import pandas as pd

# Load the CSV Sheet
df = pd.read_csv("preprocessed_telco_churn.csv")

# Remove rows with missing values in these columns
df = df.dropna()

# Selecting only the variables that are going to be used
IN = df[["Contract_One year", "Contract_Two year", "tenure", "MonthlyCharges"]]


# This is the output of what we're trying to predict
OUT = df["Churn"]

#------------------------------------------------- Splitting into train-test -------------------------------------------

# A tool that splits the data into a training and testing set respectively
from sklearn.model_selection import train_test_split

IN_train, IN_test, OUT_train, OUT_test = train_test_split(
    IN, OUT,
    test_size = 0.2, # Tests only 20% of the values, while the remaining 80% is used for training it
    random_state = 36, # Makes the split repeatable
    stratify = OUT # Splitting the data evenly
)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(IN_train, OUT_train)

pred = model.predict(IN_test)  # Predicts if it would churn based on the data


#----------------------------------------------- Displaying results ----------------------------------------------------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

acc = accuracy_score(OUT_test, pred)
prec = precision_score(OUT_test, pred, pos_label = 1)
rec = recall_score(OUT_test, pred, pos_label = 1)
f1_score = f1_score(OUT_test, pred, pos_label = 1)
cm = confusion_matrix(OUT_test, pred) # Creates Confusion Matrix

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0 (No Churn)", "Actual 1 (Churn)"],
    columns=["Predicted 0 (No Churn)", "Predicted 1 (Churn)"]
) # Properly formated the matrix 

# acc - Accuracy (How accurate was it)
# prec - Precision (How often it was correct guessing YES)
# rec - Recall (How many actual YES churners did the model catch)
# F1 Score - Measures the balance between catching real churners and avoiding false alarms

print("Logistic Regression performance:")
print("Accuracy: ", (round(acc, 1) * 100), "%")
print("Precision: ", (round(prec, 1) * 100), "%")
print("Recall: ", (round(rec,1)*100), "%")
print("F1 Score: ", (round(f1_score,1)*100), "%")
print("\nConfusion Matrix:")
print(cm_df)

#--------------------------------------------- Save model for UI -------------------------------------------------------
import joblib
joblib.dump(model, "logreg_model.pkl")


