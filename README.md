# Diabetic Predictive Model

A machine learning project for diabetes prediction and data analysis, built with Python and Streamlit. It has two separate apps:

| App | File | Purpose | Algorithm |
|-----|------|---------|-----------|
| **Diabetes Prediction System** | `myapp.py` | Predicts whether a patient is diabetic, with login, PDF report and prediction history | **Support Vector Machine (SVM)** |
| **Data Analysis & Model Evaluation** | `View_data.py` | Explores the dataset and evaluates a model with visualizations | **Random Forest Classifier** |

## Dataset

`Database/diabetes.csv` has 8 medical input features and one target column:

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Blood pressure (mm Hg) |
| SkinThickness | Skin fold thickness (mm) |
| Insulin | Insulin level |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Family history based diabetes score |
| Age | Age in years |
| **Outcome** | Target: `1` = Diabetic, `0` = Not diabetic |

## App 1: Diabetes Prediction System (`myapp.py`)

### Features

- **User authentication**: register and login system; accounts are stored locally in `users.json`
- **Prediction**: enter patient details in the sidebar and get the result (Diabetic / Not Diabetic) with its probability
- **PDF report**: download a formatted patient screening report (generated with ReportLab)
- **Prediction history**: every prediction is saved to `prediction_history.csv`, which can be downloaded as CSV
- **Model accuracy**: training and testing accuracy shown in the sidebar

### How the model works

1. Data is loaded from `Database/diabetes.csv`
2. Features are standardized with `StandardScaler`
3. Data is split 80/20 into training and test sets (`random_state=1`)
4. A **linear-kernel SVM** is trained, wrapped in `CalibratedClassifierCV` to get probability scores
5. The input from the sidebar is scaled the same way and passed to the model for prediction

## App 2: Data Analysis & Model Evaluation (`View_data.py`)

### Features

- Dataset preview, info and summary statistics
- **Data cleaning**: zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` are replaced with the column mean
- **Visualizations**:
  - Correlation heatmap
  - Glucose distribution
  - Diabetes outcome count
  - Top 8 feature importances
- **Model evaluation**: confusion matrix, classification report and accuracy score

### How the model works

1. Data is cleaned (zeros replaced with column mean)
2. Data is split 80/20 into training and test sets (`random_state=42`)
3. A **Random Forest Classifier** (100 trees) is trained
4. The model is evaluated on the test set, and feature importances show which inputs matter most for prediction

## Tech Stack

- **Language**: Python
- **Web app**: Streamlit
- **ML**: scikit-learn (SVM, Random Forest)
- **Data and plots**: pandas, NumPy, Matplotlib, Seaborn
- **Reports and images**: ReportLab, Pillow

## Project Structure

```
.
├── .devcontainer/          # Dev container configuration
├── Database/
│   └── diabetes.csv        # Dataset
├── images/                 # Images used by the project
├── myapp.py                # Prediction app (SVM)
├── View_data.py            # Data analysis app (Random Forest)
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/arpitatandon-ds/Diabetic_Predictive_Model.git
cd Diabetic_Predictive_Model
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the apps

Prediction app (SVM):

```bash
streamlit run myapp.py
```

Data analysis app (Random Forest):

```bash
streamlit run View_data.py
```

Run both commands from the project root folder. The app opens in your browser at `http://localhost:8501`.

## Usage

**Prediction app**
1. Register a new account, then log in
2. Enter the patient's details in the sidebar
3. Click **Predict** to see the result and probability
4. Download the PDF report, or use **Download History** to get all past predictions

**Analysis app**
1. Scroll through the dataset overview and charts
2. Check the confusion matrix, classification report and feature importance at the bottom

## Notes

- `users.json` and `prediction_history.csv` are created automatically when you register and predict. They are excluded from the repository through `.gitignore`.
- Passwords in this demo are stored as plain text in `users.json`. For a real-world application, they should be hashed.
- This project is for learning purposes. It is **not a medical diagnosis tool**; always consult a qualified doctor.

## Author

**Arpita Tandon**
GitHub: [@arpitatandon-ds](https://github.com/arpitatandon-ds)
