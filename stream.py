import time
from random import random as rand
from pylsl import StreamInfo, StreamOutlet
import pandas as pd


def create_stream_from_csv(csv_df, type, sampling_freq, dtype='float32', stream_prefix='stream1'):
    header = list(csv_df.columns)
    n_channels = len(header)
    info = StreamInfo('CSV', type, n_channels, sampling_freq, dtype, stream_prefix)
    desc = info.desc()
    chns = desc.append_child('channels')
    for channel in header:
        chn = chns.append_child('channel')
        chn.append_child_value('label', channel)
        # Need unit functionality
    return StreamOutlet(info)


directory = 'EMG_data_for_gestures-master/'
EMGCSV = pd.read_csv(directory + '01/1_raw_data_13-12_22.03.16.txt', sep='\t')

fs = 1000
CSVStream = create_stream_from_csv(EMGCSV, type='EMG', sampling_freq=1000)

EMGCSV = EMGCSV.astype('float32')
print(EMGCSV.dtypes)

print("now sending data...")
for idx, sample in EMGCSV.iterrows():
    #print(sample.values)
    if (sample.values[5] < -213748364) or (sample.values[5] > 2147483647):
        print('Here')
    CSVStream.push_sample(sample.values)
    time.sleep(1/fs)

print('End of CSV Reached, Closing Stream...')