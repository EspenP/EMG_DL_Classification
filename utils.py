import pandas as pd
from pandas import Timestamp
from datetime import datetime
from keras.utils import to_categorical
import scipy as sp
import scipy.signal
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from scipy.stats import moment
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from pylsl import StreamInlet

def obtain_stream_channel_names(stream):
    header = []
    inlet = StreamInlet(stream)
    info = inlet.info()
    ch = info.desc().child("channels").child("channel")
    for k in range(info.channel_count()):
        #print("  " + ch.child_value("label"))
        header.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return header

def format_data_into_dataframe(samples, timestamps, header):
    if len(header) > 0:
        df = pd.DataFrame(columns=header)
    else:
        df = pd.DataFrame()
    for sample, timestamp in zip(samples, timestamps):
        converted_time = datetime.fromtimestamp(timestamp)
        current_time = Timestamp(0).now()
        sample.append(converted_time)
        df.at[current_time] = sample
    return df


# From https://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]
 
def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values

def extract_channel_features (channel_data, fs): # Channel data should be a 1D array
    chan_feats = dict()
    # Time domain features
    chan_feats['mean'] = np.mean(channel_data)
    chan_feats['variance'] = np.var(channel_data)
    chan_feats['rms'] = np.sqrt(np.mean(channel_data**2))
    chan_feats['second_moment'] = moment(channel_data, moment=2)
    
    # PSD features
    f, P = sp.signal.welch(channel_data, fs, 'flattop', nperseg=2001, scaling='spectrum')
    try:
        max_ind = np.argmax(P)
    except Exception as e:
        print(e)
        max_ind = 0
    chan_feats['peak_frequency'] = f[max_ind]
    chan_feats['power_max'] = np.sqrt(P[max_ind])
    # chan_feats['power_mean'] = np.mean(P)

    # Frequency domain features
    N = 2001
    T = N / 1000
    yf = fft(channel_data)
    yf = 2.0/N * np.abs(yf[0:N//2])
    xf = fftfreq(N, T)[:N//2]
    ind = np.argmax(yf)
    chan_feats['max_freq'] = xf[ind]
    chan_feats['fft_peak'] = np.sqrt(yf[ind])

    t_values, autocorr_values = get_autocorr_values(channel_data, T, N, fs)
    min_ind = np.argmin(autocorr_values)
    autocorr_values = autocorr_values[min_ind:]
    ind = np.argmax(autocorr_values)
    chan_feats['max_corr'] = autocorr_values[ind]
    chan_feats['corr_time'] = t_values[ind]
    
    return chan_feats
    
def extract_window_features(df, channels = ['channel' + str(i + 1) for i in range(8)], fs=1000):
    window_feats = dict()
    df_cop = df.copy()
    for chan in channels:
        chan_data =  df[chan].values
        chan_feats = extract_channel_features(chan_data, fs)
        keys = list(chan_feats.keys())
        for key in keys:
            window_feats[chan + '_' + key] = chan_feats[key]
            
    return pd.DataFrame(window_feats, index=[0])

class ml_pipeline:

    def __init__(self):
        self.pipeline = []

    def add(self, function):
        self.pipeline.append(function)

    def predict(self, x):
        for function in self.pipeline:
            x = function(x)
            #print(x.shape)

        return x


class emg_clf:

    def __init__(self):
        self.model = self.prediction_model()

    def prediction_model(self):
        # We can now create our keras model for our simple CNN architecture
        # create model
        model = Sequential()
        model.add(LSTM(units=256,
                      dropout=0.3,
                      recurrent_dropout=0.3,
                      activation = 'tanh', 
                      name = 'layer0', 
                      return_sequences=False, 
                      input_shape=(1, 80)))

        model.add(Dense(16, activation='relu'))

        model.add(Dense(7, name='output', activation='softmax')) # Need activation function
        return model

    def load_model(self, model_name):
        self.model = tensorflow.keras.models.load_model(model_name)

    def format_data_for_lstm(self, window): # Confidence level will cause any windows 
        channels = ['channel' + str(i + 1) for i in range(8)] # where the mode count is less than 0.7 * size                                                         
        # Put windowed dfs into a 3d array                    # to be thrown out
        X = list()
        df = window.copy()
        data = df.values
        X.append(data)
        X = np.array(X)
        return X


def filter_all_channels(filt_emg, emg_keys=['channel' + str(i) for i in range(1, 9)]):
    return filt_emg[emg_keys].apply(filteremg)


def filteremg(emg, low_pass=10, sfreq=1000, high_band=20, low_band=450):
    """
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    """
    # normalise cut-off frequencies to sampling frequency
    high_band = high_band / (sfreq / 2)
    low_band = low_band / (sfreq / 2)

    # create bandpass filter for EMG
    b1, a1 = sp.signal.butter(4, [high_band, low_band], btype='bandpass')

    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(b1, a1, emg)

    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)

    # create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = low_pass / (sfreq / 2)
    b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
    emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified)

    return emg_envelope