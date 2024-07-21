from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import joblib

# load in dataset 
plotdf = pd.read_csv("data/RTADataset.csv")
# update Time variable (for plotting)
plotdf["Time"] = pd.to_datetime(plotdf['Time']) 


#load in cleaned version of dataset for classification (only for input selection)
classdf = pd.read_csv("data/cleaned.csv")

# create list for each input of classification model
age_band_list = classdf["Age_band_of_driver"].unique().tolist()
Sex_list = classdf["Sex_of_driver"].unique().tolist()
Educational_list = classdf["Educational_level"].unique().tolist()
Driving_experience_list = classdf["Driving_experience"].unique().tolist()
Lanes_or_Medians_list = classdf["Lanes_or_Medians"].unique().tolist()
Types_of_Junction_list = classdf["Types_of_Junction"].unique().tolist()
Road_surface_type_list = classdf["Road_surface_type"].unique().tolist()
Light_conditions_list = classdf["Light_conditions"].unique().tolist()
Weather_conditions_list = classdf["Weather_conditions"].unique().tolist()
Type_of_collision_list = classdf["Type_of_collision"].unique().tolist()
Vehicle_movement_list = classdf["Vehicle_movement"].unique().tolist()
Pedestrian_movement_list = classdf["Pedestrian_movement"].unique().tolist()
Cause_of_accident_list = classdf["Cause_of_accident"].unique().tolist()


# load model from disk
model = joblib.load("models/RandForModel.joblib")



app_ui = ui.page_sidebar(
    ui.sidebar(
        "Data Visualization Options",
        ui.input_checkbox_group("plotFreqOptions", "Plot Accident Severity", {"slight": "Slight", "serious": "Serious", "fatal": "Fatal"}),
        ui.input_slider("slider", "Plot Sampling Interval (min)", 3, 60, 30),
    ),

    ui.navset_tab(
        ui.nav_panel("Data Visualization",
            ui.page_fluid(
                ui.output_plot("plot"),
                ui.output_plot("pieplot"),
            ),
        ),
        ui.nav_panel("Classification",
                
                    ui.card(  
                    # ui.card_header("Card with sidebar"),
                    ui.layout_sidebar(  
                        ui.sidebar(
                            # list of inputs for classification
                            "Accident Classification Inputs",
                            ui.input_selectize("age", "Driver Age Band", age_band_list , selected="18-30"),
                            ui.input_selectize("sex", "Driver Sex", Sex_list , selected="Male"),
                            ui.input_selectize("education", "Driver Educational Level", Educational_list , selected="Junior high school"),
                            ui.input_selectize("driving_experience", "Driver Experience", Driving_experience_list , selected="5-10yr"),
                            ui.input_selectize("lane_or_median", "Lanes or Medians", Lanes_or_Medians_list , selected="Unknown"),
                            ui.input_selectize("junction_type", "Type of Junction", Types_of_Junction_list , selected="Y Shape"),
                            ui.input_selectize("road_surface", "Type of Road Surface", Road_surface_type_list , selected="Asphalt roads"),
                            ui.input_selectize("light_condition", "Light Conditions", Light_conditions_list , selected="Daylight"),
                            ui.input_selectize("weather_condition", "Weather Conditions", Weather_conditions_list , selected="Normal"),
                            ui.input_selectize("collision_type", "Type of Collision", Type_of_collision_list , selected="Vehicle with vehicle collision"),
                            ui.input_selectize("vehicle_movement", "Vehicle Movement", Vehicle_movement_list , selected="Going straight"),
                            ui.input_selectize("pedestrian_movement", "Pedestrian Movement", Pedestrian_movement_list , selected="Not a Pedestrian"),
                            ui.input_selectize("cause_accident", "Cause of Accident", Cause_of_accident_list , selected="Changing lane to the left"),
                            
                            ui.input_action_button("run_class", "Run Classification"),
                                    bg="#f8f8f8",
                                    position="right",
                                    # width=
                    ),  
                        "Classification Output:",  
                        ui.output_text_verbatim("classify"),
                    ),  
            ) 
        ),
       ui.nav_panel("Info",
                    ui.markdown(
                        """
                        **Project: Visualization and Severity Classification of Road Traffic Accidents**

                        **Purpose:**

                        In the US alone, there are over 6 million traffic accidents that occur each year, which results in over 40,000 deaths.
                        This project aims to better understand the factors that go into the severity of car/traffic accident, as well as in better 
                        visualizing data related to traffic accidents. 
                        Having such insights can go into making choices surrounding cars and the road to reduce the severity of such accidents and keeping the public safe.

                        **Context:**

                        For some context relating to the dataset and domain.
                        The dataset contains a list of traffic accidents that occured in Addis Ababa City, Ethiopia. 
                        The dataset includes a variety of collected information related to driver (Age, Driving Experience, etc), the enviroment (weather, road layout, etc), and the incident itself (Cause of incident, Severity).

                        The model trained in this project uses several of these factors to categorize the severity of the road incident. The classification of accident/injury severity is given as follows : "Fatal Injury", "Serious Injury", and "Slight Injury". For some context surrounding the dataset's content, there was a severe imbalance in the number of accidents pertaining to "Fatal" and "Serious" (in relation to "Slight"), and as a result I chose to use a technique to fix this imbalance for training the model through SMOTE. This imbalance fix is only for the data used in training, not the data visualized.


                        **Setup:**

                        For setting up the product, the general outline is very similair to that of MP1. After installing the required packages listed in requirements, you should run "app.py", which should launch the shiny app. The associated model will be loaded within "app.py", so there will be no need to train/load the model (and interact with their associated file). The listed model files are to reference of how the model was trained, and do no need to be run. Following this, all interaction can be seen on the loaded shiny page.

                        (Python Version: 3.11)

                        **Usage:**

                        Data Visualization:

                        The Data visualization tab displays two plots, a line chart displaying the frequency of accidents across the day and a pie chart showing the most common listed reasons that caused the traffic accident. The sidebar on the left handside include options to interact with the plot chart, specifically the "Plot Accident Severity" options allow you to break down the frequency of accidents based on their associated severity. The plot sample interval allows you to change how often the plot resampled (basically changing interval of how spread apart the points are based on minutes)

                        Classification:

                        The classification tab includes several inputs used to classify the severity of an accident. These inputs are then used by the model to predict and classify the severity.

                        The inputs/features are as follows (each with several associated values)
                        - Driver Age Band: Range of Driver Age
                        - Driver Sex
                        - Driver Education Level: Highest Education Level of driver
                        - Driver Experience : Years of Driving Experience
                        - Lanes or Medians: Type of Roads (Medians have a middle divider)
                        - Type of Junction
                        - Type of Road Surface
                        - Light Conditions
                        - Weather Conditions
                        - Type of Collision
                        - Vehicle Movement: Direction/Way in which vehicle was traveling
                        - Pedestrian Movement
                        - Cause of Accident: Determine cause of accident

                        Pressing the Run Classification Button at the bottom of the input sidebar after selecting your inputs, will display the models classification of the severity: "Fatal Injury / Accident",  "Serious Injury / Accident", "Slight Injury / Accident".


                        Note: The model has high tendecy to classify most inputs as "Slight". (Due to fewer data points relating to it). Some of the more influential feature include "Cause of Accident" and "Type of Collision".

                        Also as an aside, the model has been scored/tested to have an ~79% accuracy on classification
                        
                        Citation:
                        Data: Bedane, Tarikwa Tesfa (2020), “Road Traffic Accident Dataset of Addis Ababa City”, Mendeley Data, V1, doi: 10.17632/xytv86278f.1

                        Author: Hamiz Anjum
                        """
                    )
           
       ) 


    ),

    title="Car Accident Severity Classification and Frequency"
)

def server(input):
    @reactive.effect
    def _():
        print(input.text())
    
    @render.plot(alt="A Scatterplot")
    def plot():
        # TODO: Change this to a filtered version of the plot (or maybe not)
        overalldf = plotdf 

        resampleperiod = f"{input.slider()}min"

        accidentFreq = overalldf.groupby(["Time"]).size()
        resampleFreq = accidentFreq.resample(resampleperiod).sum() # can add an input to update the frequency amount
        # actualplotting

        fig, ax = plt.subplots() 
        plt.plot(resampleFreq.index, resampleFreq.values, label="All")


        if ("slight" in input.plotFreqOptions()):
            slightdf = overalldf[overalldf["Accident_severity"] == "Slight Injury"]
            slightFreq = slightdf.groupby(["Time"]).size()
            resampleSlight = slightFreq.resample(resampleperiod).sum()
            plt.plot(resampleSlight.index, resampleSlight.values, color="green", label="Slight")
        if ("serious" in input.plotFreqOptions()):
            seriousdf = overalldf[overalldf["Accident_severity"] == "Serious Injury"]
            seriousFreq = seriousdf.groupby(["Time"]).size()
            resampleSerious = seriousFreq.resample(resampleperiod).sum()
            plt.plot(resampleSerious.index, resampleSerious.values, color="orange", label="Serious")
        if ("fatal" in input.plotFreqOptions()):
            fataldf = overalldf[overalldf["Accident_severity"] == "Fatal injury"]
            fatalFreq = fataldf.groupby(["Time"]).size()
            resampleFatal = fatalFreq.resample(resampleperiod).sum()
            plt.plot(resampleFatal.index, resampleFatal.values, color="red", label="Fatal")
        
        ax.yaxis.grid(color='lightgray')


        ax.set(xlabel="Time of Day", ylabel="Number of Accidents", title="Frequency of Accidents across Day")

        ax.legend()


        

        myFmt = mdates.DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(myFmt)


        return fig
    
    @render.plot(alt="Cause of Accident")
    def pieplot():
        causes = plotdf.groupby("Cause_of_accident").size()
        for s,v in causes.items():
            # print(s,v)
            if (v < 200):
                causes["Other"] += v
                causes = causes.drop(s)
        fig, ax = plt.subplots()
        ax.pie(causes.values ,labels=causes.index, textprops={'size': 'smaller'}, autopct='%1.1f%%')
        ax.set(title="Causes of Accidents")

        return fig

    # classification model stuff
    @render.text()
    @reactive.event(input.run_class)
    def classify():
        input_data = {
            'Age_band_of_driver': [input.age()],
            'Sex_of_driver': [input.sex()], 
            'Educational_level': [input.education()],
            'Driving_experience': [input.driving_experience()], 
            'Lanes_or_Medians': [input.lane_or_median()], 
            'Types_of_Junction': [input.junction_type()],
            'Road_surface_type': [input.road_surface()], 
            'Light_conditions': [input.light_condition()], 
            'Weather_conditions' : [input.weather_condition()],
            'Type_of_collision': [input.collision_type()], 
            'Vehicle_movement': [input.vehicle_movement()], 
            'Pedestrian_movement': [input.pedestrian_movement()],
            'Cause_of_accident' : [input.cause_accident()],
        }

        inputdf = pd.DataFrame(data=input_data)

        modelclassification = model.predict(inputdf)

        # return f"{modelclassification[0]}"

        modeldict = {
            0 : "Fatal Injury / Accident",
            1 : "Serious Injury / Accident",
            2 : "Slight Injury / Accident",
        }
        return modeldict[modelclassification[0]]



app = App(app_ui, server)