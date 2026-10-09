# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
import streamlit as st
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load the dataset
df = pd.read_csv(r'Database\diabetes.csv')

# ---------------- STREAMLIT APP ---------------- #
st.title("🩺 Diabetes Prediction & Data Analysis")

# Show dataset
st.subheader("📊 Dataset Preview")
st.write(df.head())
    
# Dataset Info
st.subheader("ℹ️ Dataset Info")
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

# Summary statistics
st.subheader("📈 Summary Statistics")
st.write(df.describe())

# Step 4: Replace zeros in certain columns with mean (excluding Outcome)
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col].mean())

# Step 5: Correlation heatmap
st.subheader("🔗 Correlation Heatmap")
fig1, ax1 = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax1)
st.pyplot(fig1)

# Glucose Distribution
st.subheader("🍬 Glucose Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(df['Glucose'], kde=True, bins=30, ax=ax2)
st.pyplot(fig2)

# Step 6: Countplot of the target variable
st.subheader("⚖️ Diabetes Outcome Count")
fig3, ax3 = plt.subplots()
sns.countplot(x='Outcome', data=df, ax=ax3)
ax3.set_title("Diabetes Outcome Count (0 = No, 1 = Yes)")
st.pyplot(fig3)

# Step 7: Splitting data into training and testing sets
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']               # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Make predictions and evaluate the model
y_pred = model.predict(X_test)

st.subheader("📊 Model Evaluation")

# Confusion Matrix
st.write("**Confusion Matrix:**")
st.write(confusion_matrix(y_test, y_pred))

# Classification Report
st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred))

# Accuracy
st.write(f"**Accuracy Score:** {accuracy_score(y_test, y_pred):.2f}")

# Step 10: Feature Importance
st.subheader("🌟 Feature Importance")
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
fig4, ax4 = plt.subplots()
feat_importances.nlargest(8).plot(kind='barh', color='teal', ax=ax4)
ax4.set_title("Top Features Important for Prediction")
ax4.set_xlabel("Feature Importance Score")
st.pyplot(fig4)
