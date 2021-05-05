import pylsl
import asyncio
import logging
from Util import util
import pandas as pd
import os
from configparser import ConfigParser

logger = logging.getLogger(__name__)

configObject = ConfigParser()
configObject.read("config.ini")
fileInfo = configObject['File Info']

def get_file_info_from_config():
    return fileInfo['directory'], fileInfo['participant_name'], fileInfo['participant_session'], fileInfo['file_name_prefix']


async def save_current_streams(directory, participant_name, participant_session, file_name_prefix):
    stream_names = []
    print("looking for streams")
    streams = pylsl.resolve_streams()
    for stream in streams:
        stream_names.append(stream.name())
        print(stream.name())
        print(util.obtain_stream_channel_names(stream))

    # Initialize Inlets and Dataframes
    inlets = []
    df_dict = {}
    path = os.path.join(directory, participant_name, participant_session)
    for stream in streams:
        inlets.append(pylsl.StreamInlet(stream))
        header = util.obtain_stream_channel_names(stream)
        header.append('Device_Time')
        df_dict[stream.name()] = pd.DataFrame(columns=header)
        # Create directory for files
        try:
            os.makedirs(os.path.join(path, stream.name()))
        except:
            logger.info('Directory ', os.path.join(path, stream.name()), 'already exists')

    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        for inlet in inlets:
            samples, timestamps = inlet.pull_chunk()
            if timestamps:
                inlet_name = inlet.info().name()
                # print(inlet_name)
                current_df = df_dict[inlet_name]
                df_temp = util.format_data_into_dataframe(samples, timestamps, current_df.columns.values.tolist())
                current_df = current_df.append(df_temp)
                df_dict[inlet_name] = current_df
                file_name = os.path.join(path, inlet_name, file_name_prefix + '_data.csv')
                hdr = False if os.path.isfile(file_name) else True
                current_df.to_csv(file_name, mode='a', index_label='Timestamp', header=hdr)
                col_names = [i for i in current_df.columns]
                df_dict[inlet_name] = pd.DataFrame(columns=col_names)


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