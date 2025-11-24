#------------------------------------------------- Retrieving the data -------------------------------------------------
import pandas as pd

# Load the Excel Sheet
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Selecting only the variables that are going to be used
df = df[["Contract", "tenure", "MonthlyCharges", "Churn"]]

# Remove rows with missing values in these columns
df = df.dropna()

# This is the input of values that we are going to use to make a prediction
IN = df[["Contract", "tenure", "MonthlyCharges"]]

# This is the output of what we're trying to predict
OUT = df["Churn"]


#------------------------------------------------- Splitting into train-test -------------------------------------------

# A tool that splits the data into a training and testing set respectively
from sklearn.model_selection import train_test_split

IN_train, IN_test, OUT_train, OUT_test = train_test_split(
    IN, OUT,
    test_size = 0.2, # Tests only 20% of the values, while the remaining 80% is used for training it
    random_state = 17, # Makes the split repeatable
    stratify = OUT
)

# Because contract outputs string values and not numeric values, it's going to result in an error with the model.
# Due to that, we're going to have to convert the contract column into numeric values.
from sklearn.preprocessing import OneHotEncoder # This tool converts text columns into 0/1 columns for each.
from sklearn.compose import ColumnTransformer # This tool converts whatever is in the columns into the numeric vectors.

# This is preprocessing the columns before it's used for the model.
preprocessor = ColumnTransformer(
    transformers = [ ("contract_FIX", OneHotEncoder(handle_unknown="ignore"), ["Contract"])], remainder="passthrough")

# The new column was renamed to whatever was appropriate and the OneHotEncoder just made it into 0 and 1s.
# The "handle_unknown = 'ignore'" just ignores any new contract type that isn't already here.
# "Contract" just tells the encoder to only do this for the contract column.
# The remainder just tells the Encoder to leave the other columns as is.

#--------------------------------------------- Logistic Regression Pipeline --------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression # Tool to train the data

logreg_model = Pipeline(steps=[("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
# "prep" runs the preprocessor data.
# "clf" run logistic regression on the processed data.
# "max_iter" is the max number of times the algorithm is allowed to try improving itself while learning.

logreg_model.fit(IN_train, OUT_train)
# This line takes all the training variables and all the output (churn) of them.
# The logistic regression algorithm runs on this data and learns from it, determining what influences churn.

OUT_pred = logreg_model.predict(IN_test)
# This line makes the Logistic regression algorithm make predictions on if it would chrun.


#----------------------------------------------- Displaying results ----------------------------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(OUT_test, OUT_pred)
prec = precision_score(OUT_test, OUT_pred, pos_label = "Yes")
rec = recall_score(OUT_test, OUT_pred, pos_label = "Yes")
f1_score = f1_score(OUT_test, OUT_pred, pos_label = "Yes")

# acc - Accuracy (How accurate was it)
# prec - Precision (How often it was correct guessing YES)
# rec - Recall (How many actual YES churners did the model catch)
# F1 Score - Measures the balance between catching real churners and avoiding false alarms

print("Logistic Regression performance:")
print("Accuracy: ", (round(acc, 2) * 100), "%")
print("Precision: ", (round(prec, 2) * 100), "%")
print("Recall: ", (round(rec,2)*100), "%")
print("F1 Score: ", (round(f1_score,2)*100), "%")

#--------------------------------------------- Save model for UI -------------------------------------------------------
import joblib
joblib.dump(logreg_model, "logreg_model.pkl")
