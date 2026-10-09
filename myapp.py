import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from PIL import Image
import datetime
import os
import io
import json
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer # type: ignore
from reportlab.lib import colors # type: ignore
from reportlab.lib.pagesizes import A4 # type: ignore
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle # type: ignore

# ---------------- Config ----------------
USER_FILE = "users.json"
HISTORY_FILE = "prediction_history.csv"
IMAGE_PATH = r'D:\Internship(yuvaintern)\week1\Project\img.jpeg' 
DATA_PATH = os.path.join("Database", "diabetes.csv")        

# ---------------- User persistence ----------------
def load_users():
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"admin": "12345"}
    return {"admin": "12345"}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=2)

# ---------------- Session State Init ----------------
if "users" not in st.session_state:
    st.session_state.users = load_users()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# ------------------ UI: Login / Register ------------------
if not st.session_state.logged_in:
    st.title("🔐 Diabetes Prediction System")
    page = st.radio("Choose Page", ["Login", "Register"])

    # ---- Login form ----
    if page == "Login":
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Login")

        if submitted:
            users = st.session_state.users
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"✅ Welcome, {username}!")
                st.rerun()  
            else:
                st.error("❌ Invalid username or password")

    # ---- Register form ----
    else:
        st.subheader("Register a New Account")
        with st.form("register_form"):
            new_user = st.text_input("Choose a username", key="reg_user")
            new_pass = st.text_input("Choose a password", type="password", key="reg_pass")
            confirm_pass = st.text_input("Confirm password", type="password", key="reg_confirm")
            registered = st.form_submit_button("Register")

        if registered:
            users = st.session_state.users
            if new_user.strip() == "" or new_pass.strip() == "":
                st.error("Username and password cannot be empty.")
            elif new_user in users:
                st.error("Username already exists! Choose another.")
            elif new_pass != confirm_pass:
                st.error("Passwords do not match!")
            else:
                users[new_user] = new_pass
                save_users(users)             
                st.session_state.users = users      
                st.success(f"✅ User '{new_user}' registered successfully! You can now login.")

# ------------------ Main App (shown only when logged in) ------------------
if st.session_state.logged_in:
    st.title(f"Welcome {st.session_state.username} to Diabetes Prediction App!")

    # Logout
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun() # allowed; Streamlit will rerun the script

    # ---------- Load dataset and train model ----------
    if not os.path.exists(DATA_PATH):
        st.error(f"Diabetes CSV not found at {DATA_PATH}. Please check path.")
        st.stop()

    diabetes_df = pd.read_csv(DATA_PATH)

    # show image (if exists)
    try:
        img = Image.open(IMAGE_PATH)
        img = img.resize((200, 200))
        st.image(img, width=200)
    except Exception:
        pass

    X = diabetes_df.drop('Outcome', axis=1)
    y = diabetes_df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(model.predict(X_train), y_train)
    test_acc = accuracy_score(model.predict(X_test), y_test)

    # ---------- PDF report generator ----------
    def generate_pdf_report(record):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        subtitle_style = ParagraphStyle('subtitle', parent=styles['Heading2'], alignment=1)

        elements.append(Paragraph("🏥 Diabetes Care Center", title_style))
        elements.append(Paragraph("Patient Diabetes Screening Report", subtitle_style))
        elements.append(Spacer(1, 20))

        patient_info = [
            ["Patient Name", record["Name"]],
            ["Patient ID", record["PatientID"]],
            ["Age", record["Age"]],
            ["Gender", record["Gender"]],
            ["Date", record["Time"]]
        ]
        table1 = Table(patient_info, colWidths=[200, 300])
        table1.setStyle(TableStyle([('BOX',(0,0),(-1,-1),1,colors.black), ('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
        elements.append(table1)
        elements.append(Spacer(1, 20))

        med_details = [
            ["Pregnancies", record["Pregnancies"]],
            ["Glucose", record["Glucose"]],
            ["Blood Pressure", record["BP"]],
            ["Skin Thickness", record["SkinThickness"]],
            ["Insulin", record["Insulin"]],
            ["BMI", record["BMI"]],
            ["Diabetes Pedigree Function", record["DPF"]]
        ]
        elements.append(Paragraph("Medical Test Details", styles['Heading2']))
        table2 = Table(med_details, colWidths=[200, 300])
        table2.setStyle(TableStyle([('BOX',(0,0),(-1,-1),1,colors.black), ('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
        elements.append(table2)
        elements.append(Spacer(1, 20))

        result_color = colors.red if record["Result"] == "Diabetic" else colors.green
        result_table = Table([
            ["Prediction Result", record["Result"]],
            ["Probability", record["Probability"]]
        ], colWidths=[200,300])
        result_table.setStyle(TableStyle([
            ('BOX',(0,0),(-1,-1),1,colors.black),
            ('GRID',(0,0),(-1,-1),0.5,colors.grey),
            ('TEXTCOLOR',(1,0),(1,0),result_color)
        ]))
        elements.append(result_table)
        elements.append(Spacer(1, 20))

        elements.append(Paragraph("Doctor's Note:", styles['Heading2']))
        note = "This is an AI-generated screening report. Please consult a certified medical professional for confirmation."
        elements.append(Paragraph(note, styles['Normal']))
        elements.append(Spacer(1, 40))

        elements.append(Paragraph("Authorized by: AI Diabetes Prediction System", styles['Normal']))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    # ---------------- Sidebar: Patient details (Patient name auto-filled) ----------------
    st.sidebar.title("Enter Patient Details")
    name = st.session_state.username
    st.sidebar.text_input("Patient Name", value=name, disabled=True)  # visible but not editable

    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    preg = st.sidebar.number_input('Pregnancies', 0, 17, 6)
    glucose = st.sidebar.number_input('Glucose', 0, 199, 148)
    bp = st.sidebar.number_input('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.number_input('Skin Thickness', 0, 99, 35)
    insulin = st.sidebar.number_input('Insulin', 0, 846, 0)
    bmi = st.sidebar.number_input('BMI', 0.0, 67.1, 33.6, format="%.1f")
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', 0.078, 2.42, 0.627, format="%.3f")
    age = st.sidebar.number_input('Age', 21, 81, 50)

    st.sidebar.markdown(f"**Training Accuracy:** {train_acc:.2f}")
    st.sidebar.markdown(f"**Testing Accuracy:** {test_acc:.2f}")

    # ---------------- Predict button ----------------
    if st.sidebar.button("Predict"):
        input_data = np.array([preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        patient_id = f"P{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        record = {
            "Name": name, "PatientID": patient_id, "Gender": gender, "Pregnancies": preg,
            "Glucose": glucose, "BP": bp, "SkinThickness": skinthickness, "Insulin": insulin,
            "BMI": bmi, "DPF": dpf, "Age": age, "Result": result, "Probability": f"{prob*100:.2f}%", "Time": timestamp
        }

        # append to history CSV
        history_df = pd.DataFrame([record])
        history_df.to_csv(HISTORY_FILE, mode="a", header=not os.path.isfile(HISTORY_FILE), index=False)

        st.info(f"{name} is predicted: {result} (Probability: {prob*100:.2f}%)")

        pdf_file = generate_pdf_report(record)
        st.download_button("📄 Download Patient Report", pdf_file, file_name=f"{name}_report.pdf", mime="application/pdf")

    # ---------------- Download history ----------------
    if st.sidebar.button("Download History"):
        if os.path.isfile(HISTORY_FILE):
            with open(HISTORY_FILE, "rb") as f:
                st.sidebar.download_button("Download CSV", f, file_name=HISTORY_FILE, mime="text/csv")
        else:
            st.sidebar.error("No history found yet.")
