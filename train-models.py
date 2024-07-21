import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from datetime import datetime 
from imblearn.over_sampling import SMOTEN
import joblib


# load data
def train_model():
    df = pd.read_csv("data/cleaned.csv")

    X = df.drop(columns=["Accident_severity"])
    y = df["Accident_severity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mod = SMOTEN(random_state=0)
    X_train_smote, y_train_smote = mod.fit_resample(X_train, y_train)

    categorical_features = ['Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
        'Driving_experience', 'Lanes_or_Medians', 'Types_of_Junction',
        'Road_surface_type', 'Light_conditions', 'Weather_conditions',
        'Type_of_collision', 'Vehicle_movement', 'Pedestrian_movement',
        'Cause_of_accident']

    categorical_transformer = Pipeline(
            steps=[
                ("Modal Imputer", SimpleImputer(strategy="most_frequent")),
                ("One-Hot Encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

    preprocessor = ColumnTransformer(
            transformers=[
                ("Categorical Transformer", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )

    pipeline = Pipeline(
            steps=[
                ("Preprocessor", preprocessor),
                # ("Classifier",  SVC(gamma='auto')),
                ("Classifier",  RandomForestClassifier(max_depth=25, random_state=0, min_samples_split=6, n_estimators=250)), 

            ]
        )
    model = pipeline.fit(X_train_smote, y_train_smote)
    joblib.dump(model, "models/RandForModel2.joblib", compress=3) #different name


# Uncomment to run/train model
# train_model()