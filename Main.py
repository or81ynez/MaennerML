#importing necessery libraries for future analysis of the dataset
import folium
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as plt
import seaborn as sns
import time
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px
import torch

model_tree = 'Tree_Model.pkl'
model = torch.load(model_tree)

#Functions 

#Tranform accelerometer and gyroscope data to one dataframe
def transform_data_acceleration(file, format):
    if format == 'json':
        df = pd.read_json(file)
    else:
        df = pd.read_csv(file)  
        
    acce = df[df['sensor'] == 'Accelerometer']
    acce.reset_index(drop=True, inplace=True)   
    acce = acce.drop(columns =['seconds_elapsed','sensor', 'relativeAltitude', 'pressure', 'altitude', 'speedAccuracy', 'bearingAccuracy', 'latitude', 'altitudeAboveMeanSeaLevel', 'bearing', 'horizontalAccuracy', 'verticalAccuracy', 'longitude', 'speed', 'version', 'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])
    acce['Magnitude_acce'] = np.sqrt(acce["x"] ** 2 + acce["y"] ** 2 + acce["z"] ** 2)
    
    gyro = df[df['sensor'] == 'Gyroscope']
    gyro.reset_index(drop=True, inplace=True)   
    gyro = gyro.drop(columns = ['seconds_elapsed','sensor', 'relativeAltitude', 'pressure', 'altitude', 'speedAccuracy', 'bearingAccuracy', 'latitude', 'altitudeAboveMeanSeaLevel', 'bearing', 'horizontalAccuracy', 'verticalAccuracy', 'longitude', 'speed', 'version', 'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])
    

    for df in [gyro, acce]:
         df.index = pd.to_datetime(df['time'], unit = 'ns',errors='ignore')
         df.drop(columns=['time'], inplace=True)
    #df_new = pd.merge(loc, gyro, suffixes=('_loc', '_gyro'), on='time')
    df_new = acce.join(gyro, lsuffix = '_acce', rsuffix = '_gyro', how = 'outer').interpolate()
   
    #df_new = pd.merge(pd.merge(loc, gyro, suffixes=('_loc', '_gyro'), on='time'), acce, suffixes=('', '_acce'), on='time')
    #df_new['Type'] = type
    
    return df_new

#Tranform location from file
def transform_data_location(file, format):
    if format == 'json':
        df = pd.read_json(file)
    else:
        df = pd.read_csv(file)   

    location = df[df['sensor'] == 'Location']
    location.reset_index(drop=True, inplace=True)
    location = location.drop(columns = ['sensor', 'z', 'y', 'x', 'relativeAltitude', 'pressure', 'version', 
                                        'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])
    
    location.index = pd.to_datetime(location['time'], unit = 'ns',errors='ignore')
    location.drop(columns=['time'], inplace=True)
    #location['Type'] = type
    return location

#Cut data into windows of 5 seconds and calculate min, max, mean and std
def create_feature_df(df):   
    min_values = df.resample('5s').min(numeric_only=True)
    max_values = df.resample('5s').max(numeric_only=True)
    mean_values = df.resample('5s').mean(numeric_only=True)
    std_values = df.resample('5s').std(numeric_only=True)
    #columns_to_drop = df.columns.difference(['Magnitude_acce','speed','x_acce', 'x_gyro','y_acce', 'y_gyro', 'z_acce', 'z_gyro','x','y','z'])
    columns_to_drop = df.columns.difference(['Magnitude_acce','x_acce', 'x_gyro','y_acce', 'y_gyro', 'z_acce', 'z_gyro','x','y','z'])
    for df in [min_values, max_values, mean_values, std_values]:
        df.drop(columns=columns_to_drop, inplace=True)
    feature_df = pd.merge(pd.merge(min_values, max_values, suffixes = ('_min', '_max'), on = 'time'), pd.merge(mean_values, std_values, suffixes = ('_mean', '_std'), on = 'time'), on = 'time')
    return feature_df

#Process data for prediction
def process_data_prediction(df):
    df = transform_data_acceleration(df)
    df_prep = create_feature_df(df)
    return df_prep

def process_data_location(df):
    df = transform_data_location(df)
    return df

#Map data 
def map_data(df):
    coords = [(row.latitude, row.longitude) for _, row in df.iterrows()]
    my_map = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=16)
    folium.PolyLine(coords, color="blue", weight=5.0).add_to(my_map)
    return my_map

#read data from url
@st.experimental_memo
def get_data() -> pd.DataFrame:
    url = "http://10.100.213.5:8000/data"
    return pd.read_json(url)

### Streamlit Area
### Page config
st.set_page_config(page_title="Mapping your Data", layout="wide", page_icon=":world_map:", initial_sidebar_state="collapsed")
bg_gradient = '''
<style>
[data-testid="stAppViewContainer"] {
background: linear-gradient(#e66465, #9198e5);
}
</style>
'''
st.markdown(bg_gradient, unsafe_allow_html=True)
###

#with open('style.css') as f:
#    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
###
st.header("Dein Standort wurde gefunden")
with st.container():
    st.write("---")
    #st.subheader("Du bist gerade auf...")
    st.write("Fügen Sie bitte ihre Datei hinzu")
    def main():
        uploaded_file = st.file_uploader("Laden Sie eine JSON Datei hoch!", accept_multiple_files=False)
        if st.button("Classify me!"):
            prediction_data =  process_data_prediction(uploaded_file)
            location_data = process_data_location(uploaded_file)

            st.subheader("Dein Fortbewegungsgraph")
            map_data(location_data)
            
            tree_predictions = model_tree.predict(prediction_data)
            st.caption("Du bist gerade auf dem " + tree_predictions + "!")
                
    if __name__ == "__main__":
        main()
    st.write("---")
