import cPickle as pickle
from datetime import datetime, timedelta
import logging
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
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


    def _remove_outliers(self, df_csv):
        """
        Remove outliers in the featuers.
        """
        logger = logging.getLogger('Model._remove_outliers()')

        for col in self.cols_features:
            logger.info( 'Working on column ' + str(col) )
            column = self.features_dict[col]
            df_csv.loc[ df_csv[col] < column.value_low, col]  = column.value_default
            df_csv.loc[ df_csv[col] > column.value_high, col] = column.value_default


    def _fill_nulls(self, df_csv):
        """
        Fill NaNs in the features.
        Use median if the column is continuous.
        Use mode if the column is categorical.
        """
        logger = logging.getLogger('Model._fill_nulls()')

        for col in self.cols_categorical:
            df_csv[col].fillna(df_csv[col].mode()[0], inplace=True)

        for col in self.cols_cont:
            df_csv[col].fillna(df_csv[col].median(), inplace=True)


    def predict(self, df_features):
        """
        Given one or more rows of features, generate a prediction from the model.
        """
        logger = logging.getLogger('Model.predict()')

        # df_features should be a DataFrame.
        logger.info( 'type(df_features) = ' + str(type(df_features)) )

        df_featuers_in = df_features.copy()
        logger.info( 'df_featuers_in =\n' + str(df_featuers_in) )

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
        self._remove_outliers(df_featuers_in)
        self._fill_nulls(df_featuers_in)

        # Handle the extra feature(s).
        df_featuers_in['fractional_busy_dashers'] = df_featuers_in['total_busy_dashers'] / df_featuers_in['total_onshift_dashers']

        # Handle infinities.
        df_featuers_in['fractional_busy_dashers'].replace([np.inf, -np.inf], 9.46e-1, inplace=True)

        # Handle NaNs.
        df_featuers_in['fractional_busy_dashers'].fillna(df_featuers_in['fractional_busy_dashers'].median(), inplace=True)
        self.cols_cont.append('fractional_busy_dashers')
        self.cols_features = self.cols_categorical + self.cols_cont

        # Check for NaNs.
        df_tmp = df_featuers_in.isnull().any()
        logger.info( 'df_tmp[ df_tmp==True ] = ' + str(df_tmp[ df_tmp==True ]) )

        # Apply the model.
        X_test = df_featuers_in[self.cols_features]
        y_pred = self.model.predict(X_test)

        return y_pred


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
    model.load_model('../notebooks/rf_simple.1521943419.pkl')

    # Load input data.
    str_file_csv = 'historical_data.csv'
    df_csv = pd.read_csv('../../../data/input/' + str_file_csv,
                         parse_dates=['created_at',
                                      'actual_delivery_time'])

    # Try to calculate the outcome variable.
    col_outcome = 'outcome_total_delivery_time'
    df_csv[col_outcome] = ( df_csv['actual_delivery_time'] - df_csv['created_at'] ) / np.timedelta64(1, 's')
    logging.info( 'df_csv.head() =\n' + str(df_csv.head()) )

    # Make a few predictions.
    df_csv_slice = df_csv.iloc[0:5]
    y_pred = model.predict( df_csv_slice )
    logging.info( ' ypred = ' + str(y_pred))

    # Compute RMSE.
    y_test = df_csv_slice['outcome_total_delivery_time']
    RMSE = np.sqrt( mean_squared_error(y_test, y_pred) )
    logger.info( 'RMSE = ' + str(RMSE) )


if __name__ == "__main__":
    main()

