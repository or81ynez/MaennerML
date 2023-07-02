#Tranform accelerometer and gyroscope data to one dataframe
def transform_data_acceleration(file, format, type):
    if format == 'json':
        df = pd.read_json(file)
    else:
        df = pd.read_csv(file)   
    acce = df[df['sensor'] == 'Accelerometer']
    acce.reset_index(drop=True, inplace=True)   
    acce = acce.drop(columns =['sensor', 'relativeAltitude', 'pressure', 'altitude', 'speedAccuracy', 'bearingAccuracy', 'latitude', 'altitudeAboveMeanSeaLevel', 'bearing', 'horizontalAccuracy', 'verticalAccuracy', 'longitude', 'speed', 'version', 'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])

    gyro = df[df['sensor'] == 'Gyroscope']
    gyro.reset_index(drop=True, inplace=True)   
    gyro = gyro.drop(columns = ['sensor', 'relativeAltitude', 'pressure', 'altitude', 'speedAccuracy', 'bearingAccuracy', 'latitude', 'altitudeAboveMeanSeaLevel', 'bearing', 'horizontalAccuracy', 'verticalAccuracy', 'longitude', 'speed', 'version', 'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])
    
    for df in [gyro, acce]:
         df.index = pd.to_datetime(df['time'], unit = 'ns',errors='ignore')
         df.drop(columns=['time'], inplace=True)
    #df_new = pd.merge(loc, gyro, suffixes=('_loc', '_gyro'), on='time')
    df_new = gyro.join(acce, lsuffix = '_gyro', rsuffix = '_acce', how = 'outer').interpolate()
   
    #df_new = pd.merge(pd.merge(loc, gyro, suffixes=('_loc', '_gyro'), on='time'), acce, suffixes=('', '_acce'), on='time')
    df_new['Type'] = type
    return df_new

#Tranform accelerometer and gyroscope data to one dataframe
def transform_data_location(file, format, type):
    if format == 'json':
        df = pd.read_json(file)
    else:
        df = pd.read_csv(file)   

    location = df[df['sensor'] == 'Location']
    location.reset_index(drop=True, inplace=True)
    location = location.drop(columns = ['sensor', 'z', 'y', 'x', 'relativeAltitude', 'pressure', 'version', 'device name', 'recording time', 'platform', 'appVersion', 'device id', 'sensors', 'sampleRateMs', 'yaw', 'qx', 'qz', 'roll', 'qw', 'qy', 'pitch'])

    
    
    location.index = pd.to_datetime(location['time'], unit = 'ns',errors='ignore')
    location.drop(columns=['time'], inplace=True)
    location['Type'] = type
    return location

#Cut data into windows of 3 minutes
def cut_into_windows(df):
    df = df.resample('3T').sum()
    # Create a new DataFrame with 'time' as the index
    pivot = pd.pivot_table(df, 
                           values=['speed'], 
                           index='time', 
                           aggfunc=('min','mean', 'max'))
    # Resample the DataFrame to 3-minute intervals and calculate the sum
    #tumbling = pivot.resample('3T').sum()
    return pivot

#Combine 3-minutes windows data into one DataFrame
def combine_into_df(dfs):
    combined_df = pd.concat([cut_into_windows(df) for df in dfs])  # Apply cut_into_window to each DataFrame and concatenate them
    combined_df.reset_index(drop=True, inplace=True)  # Reset the index of the combined DataFrame
    return combined_df