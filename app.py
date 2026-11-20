
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>
.stApp {
    background-color: #f3f6fc;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #17365d;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dce5f1;
}
.stButton button {
    background-color: #2457a7;
    color: white;
    border-radius: 9px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.title("🎓 Student Performance Prediction")
st.markdown("### Machine Learning Based Academic Performance Analysis")
st.write(
    "An intelligent system that predicts student final marks "
    "using academic and personal information."
)
st.divider()

# LOAD AND PREPARE DATA
@st.cache_data
def load_data():
    df = pd.read_csv("students.csv")
    df = df.drop_duplicates().copy()

    mappings = {
        "gender": {
            "Male": 0, "Female": 1, "Other": 2
        },
        "internet_access": {
            "Yes": 1, "No": 0
        },
        "extracurricular_activities": {
            "Yes": 1, "No": 0
        },
        "parental_education": {
            "School": 0,
            "No formal education": 1,
            "Undergraduate": 2,
            "Postgraduate": 3
        }
    }

    for column, mapping in mappings.items():
        df[column] = df[column].map(mapping)

    df = df.dropna().copy()
    return df

try:
    data = load_data()

    X = data.drop(columns=["student_id", "final_marks"])
    y = data["final_marks"]

    # MODEL TRAINING
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # SIDEBAR
    st.sidebar.title("🎓 Student Predictor")
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Predict Marks", "Model Performance", "Dataset Information", "About Project"]
    )

    if page == "Home":
        st.header("Welcome to Student Performance Prediction")
        st.write(
            "This application uses Machine Learning to estimate "
            "a student's final academic marks based on different factors."
        )

        a, b, c = st.columns(3)
        a.metric("Dataset Records", f"{len(data):,}")
        b.metric("Input Features", len(X.columns))
        c.metric("ML Algorithm", "Linear Regression")

        st.subheader("How It Works")
        st.write("1. Enter student academic details.")
        st.write("2. The system processes the entered information.")
        st.write("3. The trained ML model predicts final marks.")
        st.write("4. View the prediction result.")

        st.info("Go to Predict Marks from the sidebar to try the application.")

    elif page == "Predict Marks":
        st.header("📝 Enter Student Details")
        st.write("Fill in the information below to predict academic performance.")

        with st.form("student_form"):
            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox(
                    "Gender", ["Male", "Female", "Other"]
                )
                age = st.number_input("Age", 10, 60, 18)
                study_hours = st.number_input(
                    "Study Hours Per Day", 0.0, 24.0, 4.0
                )
                attendance = st.number_input(
                    "Attendance (%)", 0.0, 100.0, 85.0
                )
                previous_marks = st.number_input(
                    "Previous Marks", 0.0, 100.0, 75.0
                )

            with col2:
                assignments = st.number_input(
                    "Assignments Completed (Out of 10)", 0, 10, 8
                )
                parental_education = st.selectbox(
                    "Parent Education",
                    ["School", "No formal education",
                     "Undergraduate", "Postgraduate"]
                )
                internet_access = st.selectbox(
                    "Internet Access", ["Yes", "No"]
                )
                extracurricular = st.selectbox(
                    "Extracurricular Activities", ["Yes", "No"]
                )

            submitted = st.form_submit_button(
                "🚀 Predict Final Marks",
                use_container_width=True
            )

        if submitted:
            gender_value = {
                "Male": 0, "Female": 1, "Other": 2
            }[gender]

            education_value = {
                "School": 0,
                "No formal education": 1,
                "Undergraduate": 2,
                "Postgraduate": 3
            }[parental_education]

            new_student = pd.DataFrame([[
                gender_value,
                age,
                study_hours,
                attendance,
                previous_marks,
                assignments,
                education_value,
                1 if internet_access == "Yes" else 0,
                1 if extracurricular == "Yes" else 0
            ]], columns=X.columns)

            new_scaled = scaler.transform(new_student)
            result = model.predict(new_scaled)[0]

            st.divider()
            st.subheader("📊 Prediction Result")

            st.metric("Predicted Final Marks", f"{result:.2f}")

            if result >= 75:
                st.success("Excellent Performance!")
            elif result >= 50:
                st.info("Good Performance!")
            else:
                st.warning("Needs Improvement")

            st.caption(
                "This is an estimated prediction from the trained model, "
                "not an actual examination result."
            )

    elif page == "Model Performance":
        st.header("📈 Model Performance Analysis")
        st.write("Evaluation of the Linear Regression model on test data.")

        a, b, c = st.columns(3)
        a.metric("MAE", f"{mae:.2f}")
        b.metric("MSE", f"{mse:.2f}")
        c.metric("R² Score", f"{r2:.3f}")

        st.subheader("Actual vs Predicted Marks")
        chart_data = pd.DataFrame({
            "Actual Marks": y_test.values[:30],
            "Predicted Marks": y_pred[:30]
        })
        st.line_chart(chart_data)

        st.write("MAE measures average prediction error.")
        st.write("MSE measures the average squared prediction error.")
        st.write("R² measures how well the model explains variation in marks.")

    elif page == "Dataset Information":
        st.header("📁 Dataset Information")

        a, b, c = st.columns(3)
        a.metric("Total Records", len(data))
        b.metric("Total Columns", len(data.columns))
        c.metric("Missing Values", int(data.isnull().sum().sum()))

        st.subheader("Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)

        st.subheader("Input Features")
        st.write(list(X.columns))

    elif page == "About Project":
        st.header("ℹ️ About This Project")

        st.write("""
        **Project Name:** Student Performance Prediction Using Machine Learning

        **Objective:** To predict students' final marks using academic
        and personal information.

        **Machine Learning Algorithm:** Linear Regression

        **Programming Language:** Python

        **Libraries Used:**
        - Pandas
        - NumPy
        - Scikit-learn
        - Streamlit

        **Dataset:** students.csv

        **Target Variable:** final_marks

        **Application:** This system provides estimated student marks
        based on the information entered by the user.
        """)

        st.success("Developed as a B.Tech Machine Learning Project.")

except FileNotFoundError:
    st.error("students.csv not found. Keep it in the same folder as app.py.")

except Exception as e:
    st.error(f"Something went wrong: {e}")
