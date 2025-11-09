#import numpy as np
import pandas as pd
#from tensorflow import keras
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_scale_data():
    data = pd.read_csv('src/data/raw/aapl_stock_data.csv')
    scaler = MinMaxScaler()
    # Exclude Date column from scaling
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data_scaled = scaler.fit_transform(data[numeric_columns])
    return data_scaled
