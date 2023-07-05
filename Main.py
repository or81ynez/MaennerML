#importing necessery libraries for future analysis of the dataset
import folium
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as plt
import seaborn as sns
import time
import numpy as np
import streamlit as st
from streamlit_lottie import st_lottie
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

#Cut data into windows of 1 minutes and calculate min, max, mean and std
def create_feature_df(df, type):   
    min_values = df.resample('1Min').min(numeric_only=True)
    max_values = df.resample('1Min').max(numeric_only=True)
    mean_values = df.resample('1Min').mean(numeric_only=True)
    std_values = df.resample('1Min').std(numeric_only=True)
    #columns_to_drop = df.columns.difference(['Magnitude_acce','speed','x_acce', 'x_gyro','y_acce', 'y_gyro', 'z_acce', 'z_gyro','x','y','z'])
    columns_to_drop = df.columns.difference(['Magnitude_acce','x_acce', 'x_gyro','y_acce', 'y_gyro', 'z_acce', 'z_gyro','x','y','z'])
    for df in [min_values, max_values, mean_values, std_values]:
        df.drop(columns=columns_to_drop, inplace=True)
    feature_df = pd.merge(pd.merge(min_values, max_values, suffixes = ('_min', '_max'), on = 'time'), pd.merge(mean_values, std_values, suffixes = ('_mean', '_std'), on = 'time'), on = 'time')
    feature_df['Type'] = type

    return feature_df

#Combine 3-minutes windows data into one DataFrame (only in case there are more than one df)
def combine_into_df(dfs, type):
    combined_df = pd.concat([create_feature_df(df, type) for df in dfs])  # Apply cut_into_window to each DataFrame and concatenate them
    #combined_df.reset_index(drop=True, inplace=True)  # Reset the index of the combined DataFrame
    return combined_df

#Map data 
def map_data(df):
    coords = [(row.latitude, row.longitude) for _, row in df.iterrows()]
    my_map = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=16)
    folium.PolyLine(coords, color="blue", weight=5.0).add_to(my_map)
    return my_map


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
    st.subheader("Now lets see, if you said the truth!")
    st.write("Now submit your data and our model will predict your mobility type. No worries, you can import json or csv files!")
    def main():
        uploaded_file = st.file_uploader("Please upload a sensor data file. JSON or .zip containing CSVs are allowed", accept_multiple_files=False)
        if st.button("Classify me!"):
            prediction_data, gps, metric_data, raw_predictions = process_data(uploaded_file)

            st.subheader("Der Ursprung deiner Daten")
            st.write("Keine Sorge, nur du kannst diese Daten sehen, wir haben nicht genug Geld für Streamlit Pro, daher können wir die nicht speichern ;D")
            st.map(gps)

            st.subheader("Dein Fortbewegungsgraph")
            output_string = ""
            
            graph = graphviz.Digraph()
            i = 0
            if len(prediction_data) > 1:
                while i < len(prediction_data) -1:
                    graph.edge((prediction_data[i][0] + " " + str(prediction_data[i][1]) + " min"), (prediction_data[i+1][0] + " " + str(prediction_data[i+1][1]) + " min"))
                    i += 1
            else:
                graph.edge(prediction_data[i][0] + " " + str(prediction_data[i][1]) + " min", "End")
            st.write(output_string)
            st.graphviz_chart(graph)

            st.subheader("Deine Fortbewegungsverteilung")
                
    if __name__ == "__main__":
        main()
    st.write("---")
