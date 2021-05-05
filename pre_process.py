import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_path = 'EMG_data_for_gestures-master/'
p1_df = pd.read_csv(data_path + '01/' + '1_raw_data_13-12_22.03.16.txt', sep='\t')

print(p1_df)