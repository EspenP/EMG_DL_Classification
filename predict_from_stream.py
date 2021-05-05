import pylsl
import asyncio
import logging

import pandas as pd
from pandas import Timestamp
from datetime import datetime, timedelta
import os
from configparser import ConfigParser
from scipy.stats import mode
from utils import filter_all_channels, emg_clf, ml_pipeline, extract_window_features
import utils

import numpy as np
import tensorflow

logger = logging.getLogger(__name__)

configObject = ConfigParser()
configObject.read("config.ini")
fileInfo = configObject['File Info']
modelInfo = configObject['Model Info']
predInfo = configObject['Prediction Info']

def get_file_info_from_config():
    return fileInfo['directory'], fileInfo['participant_name'], fileInfo['participant_session'], fileInfo['file_name_prefix']


async def save_current_streams(directory, participant_name, participant_session, file_name_prefix):
    stream_names = []
    print("looking for streams")
    streams = pylsl.resolve_streams()
    for stream in streams:
        stream_names.append(stream.name())
        print(stream.name())
        print(utils.obtain_stream_channel_names(stream))

    # Initialize Inlets and Dataframes
    inlets = []
    df_dict = {}
    path = os.path.join(directory, participant_name, participant_session)
    for stream in streams:
        inlets.append(pylsl.StreamInlet(stream))
        header = utils.obtain_stream_channel_names(stream)
        header.append('Device_Time')
        df_dict[stream.name()] = pd.DataFrame(columns=header)
        # Create directory for files
        try:
            os.makedirs(os.path.join(path, stream.name()))
        except:
            logger.info('Directory ', os.path.join(path, stream.name()), 'already exists')


    # Initialize Model and Pipeline
    clf = emg_clf()
    clf.load_model(modelInfo['directory'] + modelInfo['model'])
    pipeline = ml_pipeline()
    pipeline.add(filter_all_channels)
    pipeline.add(extract_window_features)
    pipeline.add(clf.format_data_for_lstm)
    pipeline.add(clf.model.predict)
    pipeline.add(np.argmax)

    window_size = int(predInfo['window'])
    stride = int(predInfo['stride'])
    input_x = int(predInfo['input_x'])

    inlet = inlets[0]
    inlet_name = inlet.info().name()
    file_name = os.path.join(path, inlet_name, file_name_prefix + '_data.csv')
    pred_name = os.path.join(path, inlet_name, file_name_prefix + '_pred.csv')

    stream_df = pd.DataFrame(columns=header)
    next_update = Timestamp(0).now() + timedelta(seconds=stride)
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        samples, timestamps = inlet.pull_chunk()
        if timestamps:

            # Save samples to csv
            sample_df = df_dict[inlet_name]
            df_temp = utils.format_data_into_dataframe(samples, timestamps, sample_df.columns.values.tolist())
            sample_df = sample_df.append(df_temp)
            df_dict[inlet_name] = sample_df
            hdr = False if os.path.isfile(file_name) else True
            sample_df.to_csv(file_name, mode='a', index_label='Timestamp', header=hdr)
            col_names = [i for i in sample_df.columns]
            df_dict[inlet_name] = pd.DataFrame(columns=col_names)

            # Append samples to the stream df
            stream_df = stream_df.append(sample_df)

            # See if we have enough samples to make a predicition on
            if len(stream_df) < input_x:
                continue
            # Save predictions every update
            if Timestamp(0).now() >= next_update:
                # get the last input_x samples
                windowed_df = stream_df.iloc[-input_x:]
                # update stream_df to manage space
                stream_df = windowed_df
                # predict using our pipeline
                prediction = pipeline.predict(windowed_df)
                # determine the true class from the windowed_df
                target = mode(windowed_df['class'].values)[0][0]
                # get the current time
                time = Timestamp(0).now()
                # print everything
                print(time, prediction, target)

                # Initialize a new dataframe and add our data to it
                pred_df = pd.DataFrame()
                pred_df['Timestamp'] = [time]
                pred_df['Prediction'] = [prediction]
                pred_df['Target'] = [target]

                # Append everything to a csv file
                pred_hdr = False if os.path.isfile(pred_name) else True
                pred_df.to_csv(pred_name, mode='a', index_label='Timestamp', header=pred_hdr)
                next_update = Timestamp(0).now() + timedelta(seconds=stride)


async def main():
    directory, participant_name, participant_session, file_name_prefix = get_file_info_from_config()
    await save_current_streams(directory, participant_name, participant_session, file_name_prefix)


if __name__ == '__main__':
    asyncio.ensure_future(main())
    loop = asyncio.get_event_loop()
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        logger.info("Ctrl-C pressed.")
    finally:
        loop.close()