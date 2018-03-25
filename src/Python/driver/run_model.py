from datetime import datetime
import json
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
import sys
import time

sys.path.append('../model/')
from model import Model


def get_input_data(filename):
    logger = logging.getLogger('get_input_data()')

    logger.info( 'Reading from ' + filename )
    # Read every line of the JSON file, and parse it into a dictionary.
    with open(filename, 'r') as f:
        lines = f.readlines()

    dict_list = []
    for line in lines:
        parsed_json = json.loads(line)
        dict_list.append( parsed_json )

    df = pd.DataFrame( dict_list )

    # Parse dates.
    cols_dates = ['created_at']
    for col in cols_dates:
        df[col] = pd.to_datetime( df[col] )

    return df


def main():
    # Configure logging.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(process)d/%(threadName)s - %(name)s - %(levelname)s - %(message)s',
                        #stream=sys.stdout)
                        filename='./run_model.log',
                        filemode='w')
    logger = logging.getLogger('main()')

    pd.set_option('display.max_columns', None)

    model = Model()
    model.load_model('../notebooks/rf_simple.1521943419.pkl')

    # Load input data.
    str_file_csv = 'data_to_predict.json'
    df_csv = get_input_data('../../../data/input/' + str_file_csv)
    logging.info( 'df_csv.head() =\n' + str(df_csv.head()) )

    # Make a few predictions.
    y_pred = model.predict( df_csv )
    logging.info( ' y_pred = ' + str(y_pred))
    df_csv['predicted_delivery_seconds'] = y_pred


    # Write output to file.
    df_csv[['delivery_id', 'predicted_delivery_seconds']].to_csv('output.tsv', sep='\t', index=False)


if __name__ == "__main__":
    main()

