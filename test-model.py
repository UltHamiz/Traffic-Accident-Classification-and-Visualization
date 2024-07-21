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


# Test Model
# NOTE: Outputs correspond as following:
# 0 : Fatal
# 1 : Serious
# 2 : Slight
def test_model():
    loadedmodel = joblib.load("models/RandForModel.joblib")

    # sample input
    input_data = {
        'Age_band_of_driver': ["31-50"],
        'Sex_of_driver': ["Male"], 
        'Educational_level': ["Junior high school"],
        'Driving_experience': ["Above 10yr"], 
        'Lanes_or_Medians': ["Undivided Two way"], 
        'Types_of_Junction': ["No junction"],
        'Road_surface_type': ["Asphalt roads"], 
        'Light_conditions': ["Daylight"], 
        'Weather_conditions' : ["Normal"],
        'Type_of_collision': ["Collision with roadside objects"], 
        'Vehicle_movement': ["Going straight"], 
        'Pedestrian_movement': ["Not a Pedestrian"],
        'Cause_of_accident' : ["Overtaking"],
    }

    mydf = pd.DataFrame(data=input_data)
    output =  loadedmodel.predict(mydf)
    print(output) #prints out model prediction

    # prints out accuracy from test split
    df = pd.read_csv("data/cleaned.csv")
    X = df.drop(columns=["Accident_severity"])
    y = df["Accident_severity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(loadedmodel.score(X_test, y_test)) #print out score