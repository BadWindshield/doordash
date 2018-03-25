import cPickle as pickle
from datetime import datetime, timedelta
import logging
import math
import numpy as np
import pandas as pd
import sys


class Feature(object):
    def __init__(self, value_low, value_high, value_default):
        self.value_low = value_low
        self.value_high = value_high
        self.value_default = value_default


class Model(object):
    def __init__(self):
        self.features_dict = { 'created_at_hour' : Feature(0, 23, 2),
                               'created_at_dayofweek' : Feature(0, 6, 5),
                               'market_id' : Feature(1.0, 6.0, 2.0),
                               'order_protocol' : Feature(1.0, 7.0, 1.0),
                               'total_items' : Feature(0, 13.0, 3.0),
                               'subtotal' : Feature(0, 9470, 2200),
                               'num_distinct_items' : Feature(0, 9.0, 2.0),
                               'min_item_price' : Feature(0.0, 2501.0, 595.0),
                               'max_item_price' : Feature(0.0, 3078.0, 1095.0),
                               'total_onshift_dashers' : Feature(0, 137.0, 37.0),
                               'total_busy_dashers' : Feature(0, 127.0, 34.0),
                               'total_outstanding_orders' : Feature(0, 214.0, 41.0),
                               'estimated_order_place_duration' : Feature(0, 447.0, 251.0),
                               'estimated_store_to_consumer_driving_duration' : Feature(109.0, 1051.0, 544.0) }

        self.cols_categorical = ['created_at_hour',
                                 'created_at_dayofweek',
                                 'market_id',
                                 'order_protocol']

        self.cols_cont = [ 'total_items',
                           'subtotal',
                           'num_distinct_items',
                           'min_item_price',
                           'max_item_price',
                           'total_onshift_dashers',
                           'total_busy_dashers',
                           'total_outstanding_orders',
                           'estimated_order_place_duration',
                           'estimated_store_to_consumer_driving_duration']

        self.cols_features = self.cols_categorical + self.cols_cont

        self.model = None


    def load_model(self, pickle_file_name):
        """
        Load a previously trained model.
        """
        logger = logging.getLogger('Model.load_model()')

        self.model = pickle.load( open( pickle_file_name, 'rb') )


    def _remove_outliers(self):
        """
        Remove outliers in the featuers.
        """
        logger = logging.getLogger('Model._remove_outliers()')


    def _fill_nulls(self):
        """
        Fill NaNs in the features.
        """
        logger = logging.getLogger('Model._fill_nulls()')


    def predict(self, df_features):
        """
        Given a row of features, generate a prediction from the model.
        """
        logger = logging.getLogger('Model.predict()')

        df_featuers_in = df_features.copy()

        try:
            # Might not exist.
            df_featuers_in['created_at_hour'] = df_featuers_in['created_at'].dt.hour
        except Exception as e:
            logger.exception( 'Caught exception ' + str(e) )

        try:
            # Might not exist
            df_featuers_in['created_at_dayofweek'] = df_featuers_in['created_at'].dt.dayofweek
        except Exception as e:
            logger.exception( 'Caught exception ' + str(e) )

        # Clean the features.
        self._remove_outliers()
        self._fill_nulls()


def main():
    # Configure logging.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(process)d/%(threadName)s - %(name)s - %(levelname)s - %(message)s',
                        #stream=sys.stdout)
                        filename='./model.log',
                        filemode='w')
    logger = logging.getLogger('main()')

    pd.set_option('display.max_columns', None)

    model = Model()
    # model.load_model('../notebook/rf.1521943419.pkl')

    # Load input data.
    str_file_csv = 'historical_data.csv'
    df_csv = pd.read_csv('../../../data/input/' + str_file_csv,
                         parse_dates=['created_at',
                                      'actual_delivery_time'])
    logging.info( 'df_csv.head() =\n' + str(df_csv.head()) )


if __name__ == "__main__":
    main()

