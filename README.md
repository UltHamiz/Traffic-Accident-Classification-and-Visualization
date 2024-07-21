# CS 498 E2E Final Project (sp24) repo for NetID: manju2

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


Note: The model has high tendecy to classify most inputs as "Slight". (Due to fewer data points relating to it). Some of the more influential feature include "Cause of Accident" and "Type of Collision" as well as driver "Education", see the default input for example of "Serious" classification

Also as an aside, the model has been scored/tested to have an ~79% accuracy on classification.

Citation:
Data: Bedane, Tarikwa Tesfa (2020), “Road Traffic Accident Dataset of Addis Ababa City”, Mendeley Data, V1, doi: 10.17632/xytv86278f.1

