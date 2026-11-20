import pandas as pd

#load the dataset
data = pd.read_csv('students.csv')

#display first five rows
print(data.head)

print(data.shape)#rows and columns
print(data.info())#data types 
print(data.isnull().sum()) #check missing values

#dataset analysis

# Display column names
print("Column Names:")
print(data.columns.tolist())

# Statistical analysis
print("\nDataset Statistics:")
print(data.describe(include="all"))

#data cleaning

# Check missing values
print("Missing Values:")
print(data.isnull().sum())

# Check duplicate rows
print("Duplicate Rows:")
print(data.duplicated().sum())

# Remove duplicate rows
data = data.drop_duplicates()

print("Dataset Shape After Cleaning:")
print(data.shape)


# Check data types
print("Data Types:")
print(data.dtypes)

# Check categorical data
print("\nCategorical Columns:")
print(data.select_dtypes(include='object').columns.tolist())

# Check numerical data
print("\nNumerical Columns:")
print(data.select_dtypes(include='number').columns.tolist())


#unique values
print(data["gender"].unique())
print(data["parental_education"].unique())
print(data["internet_access"].unique())
print(data["extracurricular_activities"].unique())


# Encoding categorical columns i.e. converting text to numbers 

data["gender"] = data["gender"].map({
    "Male": 0,
    "Female": 1,
    "Other": 2
})


data["internet_access"] = data["internet_access"].map({
    "Yes": 1,
    "No": 0
})



data["extracurricular_activities"] = data["extracurricular_activities"].map({
    "Yes": 1,
    "No": 0
})

print(data.head())


# Ordinal Encoding i.e. encoding parental_education


#dictionary
education_mapping = {    
    "School": 0,
    "No formal education": 1,
    "Undergraduate": 2,
    "Postgraduate": 3
}

data["parental_education"] = data["parental_education"].map(education_mapping)

print(data["parental_education"].unique())
print(data.head())



# Feature Selection


# input features
X = data.drop(columns=["student_id", "final_marks"])

# target feature
y = data["final_marks"]

print("Input Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


from sklearn.model_selection import train_test_split

# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# feature Scaling

from sklearn.preprocessing import MinMaxScaler

# Create MinMaxScaler
scaler = MinMaxScaler()

# Scale training data
X_train_scaled = scaler.fit_transform(X_train)

# Scale testing data
X_test_scaled = scaler.transform(X_test)

print("Training Data Shape:", X_train_scaled.shape)
print("Testing Data Shape:", X_test_scaled.shape)


# Model training


from sklearn.linear_model import LinearRegression

# Create model
model = LinearRegression()

# Train model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

print("Predicted Marks:")
print(y_pred[:10])



# Model Evaluation

# MAE
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", mae)

# MSE
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)

# R² score
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)



# Trying 2nd Model ie " RANDOM FOREST "


from sklearn.ensemble import RandomForestRegressor

# Create Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

print("Random Forest Predictions:")
print(rf_pred[:10])



# Model Evaluation


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest Evaluation")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("R2 Score:", rf_r2)




# Feature importance

import pandas as pd

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("Feature Importance:")
print(importance)



# graph

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Student Performance Prediction")

plt.gca().invert_yaxis()

plt.show()







# Model training and evaluation done







# New performance Prediction

new_student = pd.DataFrame(
    [[1, 18, 4, 85, 75, 8, 2, 1, 1]],
    columns=X.columns
)

''' Feature	Example value
Gender	1 (Female)
Age	18
Study Hours	4
Attendance	85%
Previous Marks	75
Assignments	8
Parent Education 2 (Undergraduate)
Internet Access	1 (Yes)
Extracurricular	1 (Yes)'''

# Apply feature scaling
new_student_scaled = scaler.transform(new_student)

# Predict final marks
prediction = model.predict(new_student_scaled)

print("New Student Predicted Marks:", prediction[0])






# Take input from user

print("\nEnter New Student Details")

gender = int(input("Gender (0=Male, 1=Female, 2=Other): "))
age = int(input("Age: "))
study_hours = float(input("Study Hours per Day: "))
attendance = float(input("Attendance Percentage: "))
previous_marks = float(input("Previous Marks: "))
assignments = int(input("Assignments Completed (out of 10): "))
parental_education = int(input("Parent Education (0=School, 1=No Formal Education, 2=Undergraduate, 3=Postgraduate): "))
internet_access = int(input("Internet Access (0=No, 1=Yes): "))
extracurricular_activities = int(input("Extracurricular Activities (0=No, 1=Yes): "))

new_student = pd.DataFrame(
    [[gender, age, study_hours, attendance,
      previous_marks, assignments, parental_education,
      internet_access, extracurricular_activities ]],
    columns=X.columns
)

new_student_scaled = scaler.transform(new_student)

prediction = model.predict(new_student_scaled)

print("\nPredicted Final Marks:", prediction[0])






# PROJECT COMPLETED
