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


    def precict(self, X):
        """
        Given a row of features, generate a prediction from the model.
        """
        logger = logging.getLogger('Model.precict()')

        # Clean the features.
        self._remove_outliers()
        self._fill_nulls()


