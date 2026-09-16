import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="House Rent Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Rent Prediction System")

st.write(
    "Enter the house details below to predict the monthly rent."
)


# ==================================================
# LOAD DATASET
# ==================================================

@st.cache_data
def load_data():

    # Find the folder where main.py is located
    base_dir = Path(__file__).resolve().parent

    # Dataset is in the same folder as main.py
    file_path = base_dir / "House_Rent_Dataset.csv"

    df = pd.read_csv(file_path)

    return df


df = load_data()


# Remove extra spaces from column names
df.columns = df.columns.str.strip()


# ==================================================
# REMOVE UNNECESSARY COLUMN
# ==================================================

if "Posted On" in df.columns:
    df = df.drop("Posted On", axis=1)


# ==================================================
# TARGET
# ==================================================

target = "Rent"

X = df.drop(target, axis=1)

y = df[target]


# ==================================================
# FIND CATEGORICAL AND NUMERICAL COLUMNS
# ==================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# ==================================================
# PREPROCESSING
# ==================================================

preprocessor = ColumnTransformer(
    transformers=[

        # Categorical columns
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        ),

        # Numerical columns
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# ==================================================
# RANDOM FOREST MODEL
# ==================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ==================================================
# CREATE PIPELINE
# ==================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# TRAIN MODEL
# ==================================================

@st.cache_resource
def train_model(X_train, y_train):

    pipeline.fit(X_train, y_train)

    return pipeline


pipeline = train_model(X_train, y_train)


# ==================================================
# HOUSE DETAILS
# ==================================================

st.header("🏡 Enter House Details")


col1, col2 = st.columns(2)


# ==================================================
# LEFT COLUMN
# ==================================================

with col1:

    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    size = st.number_input(
        "Size (sq ft)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=50
    )

    bathroom = st.number_input(
        "Bathroom",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    floor = st.selectbox(
        "Floor",
        sorted(
            df["Floor"]
            .dropna()
            .astype(str)
            .unique()
        )
    )

    area_type = st.selectbox(
        "Area Type",
        sorted(
            df["Area Type"]
            .dropna()
            .astype(str)
            .unique()
        )
    )


# ==================================================
# RIGHT COLUMN
# ==================================================

with col2:

    area_locality = st.selectbox(
        "Area Locality",
        sorted(
            df["Area Locality"]
            .dropna()
            .astype(str)
            .unique()
        )
    )

    city = st.selectbox(
        "City",
        sorted(
            df["City"]
            .dropna()
            .astype(str)
            .unique()
        )
    )

    furnishing_status = st.selectbox(
        "Furnishing Status",
        sorted(
            df["Furnishing Status"]
            .dropna()
            .astype(str)
            .unique()
        )
    )

    tenant_preferred = st.selectbox(
        "Tenant Preferred",
        sorted(
            df["Tenant Preferred"]
            .dropna()
            .astype(str)
            .unique()
        )
    )

    point_of_contact = st.selectbox(
        "Point of Contact",
        sorted(
            df["Point of Contact"]
            .dropna()
            .astype(str)
            .unique()
        )
    )


# ==================================================
# CREATE INPUT DATA
# ==================================================

input_data = pd.DataFrame({

    "BHK": [bhk],

    "Size": [size],

    "Floor": [floor],

    "Area Type": [area_type],

    "Area Locality": [area_locality],

    "City": [city],

    "Furnishing Status": [furnishing_status],

    "Tenant Preferred": [tenant_preferred],

    "Bathroom": [bathroom],

    "Point of Contact": [point_of_contact]
})


# ==================================================
# PREDICT RENT
# ==================================================

st.write("")


if st.button(
    "🔮 Predict Rent",
    use_container_width=True
):

    prediction = pipeline.predict(input_data)[0]

    st.success(
        f"💰 Predicted Monthly Rent: ₹ {prediction:,.0f}"
    )