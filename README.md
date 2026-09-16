# 🏠 House Rent Prediction System

A beginner-friendly **Streamlit web application** that predicts the monthly rent of a house using a **Random Forest Regressor**.

## 📌 Project Overview

This project takes house details such as BHK, size, floor, area type, locality, city, furnishing status, tenant preference, bathroom count, and point of contact, and predicts the expected monthly rent.

The application uses:
- **Pandas** for loading and handling the dataset
- **Scikit-learn** for preprocessing, model training, and prediction
- **Random Forest Regression** for rent prediction
- **Streamlit** for the web interface

## 🧠 How It Works

1. The application loads `House_Rent_Dataset.csv`.
2. Extra spaces are removed from column names.
3. The `Posted On` column is removed because it is not used as a prediction feature.
4. `Rent` is used as the target variable.
5. Categorical features are converted using `OneHotEncoder`.
6. Numerical features are passed through without transformation.
7. The data is divided into training and testing sets.
8. A Random Forest Regressor is trained on the training data.
9. The user enters house details in the Streamlit interface.
10. The trained model predicts the monthly rent.

## 📂 Project Structure

```text
House-Rent-Prediction/
│
├── main.py
├── House_Rent_Dataset.csv
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone this repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Run the following command:

```bash
streamlit run main.py
```

The application will open in your web browser.

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn

## 🤖 Machine Learning Model

The project uses:

**RandomForestRegressor**

Main model settings:
- `n_estimators=200`
- `random_state=42`
- `n_jobs=-1`

The model uses a Scikit-learn pipeline containing preprocessing and the Random Forest model.

## 📊 Input Features

The application uses the following house details:

- BHK
- Size
- Floor
- Area Type
- Area Locality
- City
- Furnishing Status
- Tenant Preferred
- Bathroom
- Point of Contact

## 🎯 Target

The target variable is:

```text
Rent
```

The application displays the predicted monthly rent after the user clicks **Predict Rent**.

## 👩‍💻 Author

Developed as a machine learning / Streamlit project.
