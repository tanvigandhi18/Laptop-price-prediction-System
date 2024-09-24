
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from google.colab import drive
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import RBFSampler
from sklearn.base import BaseEstimator, TransformerMixin
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler


def preprocessing(df):
    df['Ram'] = df['Ram'].str[0:-2].astype(int)   
    df['Weight'] = df['Weight'].str[0:-2].astype(float)

    
    opsys_data = df[['OpSys']]                         #One-hot
    encoder = OneHotEncoder(drop='first')
    #Fit and transforming the 'OpSys' data to get a dense matrix
    encoded_opsys = encoder.fit_transform(opsys_data).toarray()
    # Converting the encoded data into a DataFrame
    encoded_opsys_df = pd.DataFrame(encoded_opsys, columns=encoder.get_feature_names_out(['OpSys']))
    
    #print(encoded_opsys_df)
    
    #Dropping the original 'OpSys' column from the original dataframe
    df_dropped = df.drop('OpSys', axis=1)
    
    #Concatenating the original dataframe (without 'OpSys') and the encoded_opsys_df
    df_2 = pd.concat([df_dropped.reset_index(drop=True),
                                   encoded_opsys_df.reset_index(drop=True)], axis=1)

    
    
    company_data = df_2[['Company']]                   #One-hot
    encoder = OneHotEncoder(drop='first')
    #Fit and transforming the 'Company' data to get a dense matrix
    encoded_company = encoder.fit_transform(company_data).toarray()
    # Converting the encoded data into a DataFrame
    encoded_company_df = pd.DataFrame(encoded_company, columns=encoder.get_feature_names_out(['Company']))
    
     
    #Dropping the original 'Company' column from the original dataframe
    df_dropped = df_2.drop('Company', axis=1)
    
    #Concatenating the original dataframe (without 'OpSys') and the encoded_opsys_df
    df_2 = pd.concat([df_dropped.reset_index(drop=True),
                                   encoded_company_df.reset_index(drop=True)], axis=1)

    
    TypeName_data = df_2[['TypeName']]                  #One-hot
    encoder = OneHotEncoder(drop='first')
    #Fit and transforming the 'TypeName' data to get a dense matrix
    encoded_TypeName = encoder.fit_transform(TypeName_data).toarray()
    # Converting the encoded data into a DataFrame
    encoded_TypeName_df = pd.DataFrame(encoded_TypeName, columns=encoder.get_feature_names_out(['TypeName']))
    
    
    
    #Dropping the original 'TypeName' column from the original dataframe
    df_dropped = df_2.drop('TypeName', axis=1)
    
    #Concatenating the original dataframe (without 'OpSys') and the encoded_opsys_df
    df_2 = pd.concat([df_dropped.reset_index(drop=True),
                                   encoded_TypeName_df.reset_index(drop=True)], axis=1)
    


    pattern = r'(?P<displayName>.*?)(?P<Resolution>\d+x\d+)$'

    # Extracting displayName and Resolution using the defined pattern
    df_2[['displayName', 'Resolution']] = df['ScreenResolution'].str.extract(pattern)
    df_2 = df_2.drop(columns=['ScreenResolution'])


    def fetch_processor(x):
      cpu_name = " ".join(x.split()[0:3])
      if cpu_name == 'Intel Core i7' or cpu_name == 'Intel Core i5' or cpu_name == 'Intel Core i3':
        return cpu_name
      elif cpu_name.split()[0] == 'Intel':
        return 'Other Intel Processor'
      else:
        return 'AMD Processor'

    df_2['CPU_Brand'] = df['Cpu'].apply(fetch_processor)
    #print(df_2[['Cpu', 'CPU_Brand']])
     


    clock_speed_pattern = r'(\d+\.?\d*\s*GHz)'
    # Extract the clock speed
    df_2['CPU_Clock_Speed'] = df_2['Cpu'].str.extract(clock_speed_pattern)
    df_2 = df_2.drop(columns=['Cpu'])
    df_2.CPU_Brand.unique()
    cpu_brand_data = df_2[['CPU_Brand']]
    encoder = OneHotEncoder(drop='first')
    encoded_cpu_brand = encoder.fit_transform(cpu_brand_data).toarray()
    encoded_cpu_brand_df = pd.DataFrame(encoded_cpu_brand, columns=encoder.get_feature_names_out(['CPU_Brand']))
    df_dropped = df_2.drop('CPU_Brand', axis=1)
    df_2 = pd.concat([df_dropped.reset_index(drop=True),
                                   encoded_cpu_brand_df.reset_index(drop=True)], axis=1)


    df_2['HDD'] = 0
    df_2['SSD'] = 0
    df_2['Flash Storage'] = 0

    # Iterate over each row
    for index, row in df.iterrows():
        memory = row['Memory']
        # Split the string by '+'
        components = memory.split('+')
        for comp in components:
            # Extract storage capacity and type
            storage = comp.strip().split(' ')
            #capacity = int(storage[0].replace('GB', '')) if 'GB' in storage[0] else int(storage[0].replace('TB', '')) * 1024
            capacity = float(storage[0].replace('GB', '')) if 'GB' in storage[0] else float(storage[0].replace('TB', '')) * 1024
            if 'HDD' in storage:
                df_2.at[index, 'HDD'] = capacity
            elif 'SSD' in storage:
                df_2.at[index, 'SSD'] = capacity
            elif 'Flash' in storage:
                df_2.at[index, 'Flash Storage'] = capacity
    
    # Drop the original 'Memory' column
    df_2.drop(columns=['Memory'], inplace=True)


    df_2['CPU_Clock_Speed'] = df_2['CPU_Clock_Speed'].str.replace("GHz", "")

    
    features = ['IPS Panel', 'Full HD', 'Touchscreen', '4K Ultra HD', 'Quad HD+', 'Retina Display']
    # Create a separate column for each feature
    for feature in features:
        df_2[feature] = df_2['displayName'].str.contains(feature, case=False, na=False).astype(int)
    # Drop the 'displayName' column from the DataFrame
    df_2 = df_2.drop(columns='displayName')

    
    df_2.Resolution.unique()
    df_2[['Width', 'Height']] = df_2['Resolution'].str.split('x', expand=True).astype(int)
    df_2 = df_2.drop(columns='Resolution')

    df_2['PPI'] = (((df_2['Width']**2) + (df_2['Height']**2))**0.5/df_2['Inches']).astype('float')
    # df_2.corr()['Price'].sort_values(ascending=False)

    df_2.drop(columns = ['Inches','Height','Width'], inplace=True)


    df_2.drop(columns='Number', inplace=True)


    # Which brand GPU is in laptop
    df_2['Gpu_brand'] = df_2['Gpu'].apply(lambda x: x.split()[0])
    #print(df_2.Gpu_brand.unique())
    df_2.drop(columns=['Gpu'],inplace=True)
    
     
    #Now applying on-hot encoding on Gpu_brand column
    
    # Extract CPU Brand data
    gpu_brand_data = df_2[['Gpu_brand']]
    encoder = OneHotEncoder(drop='first')
    encoded_gpu_brand = encoder.fit_transform(gpu_brand_data).toarray()
    # Converting the encoded data into a DataFrame
    encoded_gpu_brand_df = pd.DataFrame(encoded_gpu_brand, columns=encoder.get_feature_names_out(['Gpu_brand']))
    df_dropped = df_2.drop('Gpu_brand', axis=1)
    # Concatenating into the original dataframe
    df_2 = pd.concat([df_dropped.reset_index(drop=True), encoded_gpu_brand_df.reset_index(drop=True)], axis=1)

    return df_2

    

    
     
