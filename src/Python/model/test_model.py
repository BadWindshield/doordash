from datetime import datetime
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import sys
import time
import unittest

from model import Model


class TestModel(unittest.TestCase):
    """
    Apply the model to a few training samples.
    """
    def setUp(self):
        logger = logging.getLogger('TestModel.setUp()')

        self.model = Model()
        self.model.load_model('../notebooks/rf_simple.1521943419.pkl')

        # Load input data.
        str_file_csv = 'historical_data.csv'
        self.df_csv = pd.read_csv('../../../data/input/' + str_file_csv,
                                  parse_dates=['created_at',
                                               'actual_delivery_time'])

        # Try to calculate the outcome variable.
        col_outcome = 'outcome_total_delivery_time'
        self.df_csv[col_outcome] = ( self.df_csv['actual_delivery_time'] - self.df_csv['created_at'] ) / np.timedelta64(1, 's')

    def tearDown(self):
        logger = logging.getLogger('TestModel.tearDown()')

    def test_predictions(self):
        logger = logging.getLogger('TestModel.test_predictions()')

        # Make a few predictions.
        num_rows = 50
        df_csv_slice = self.df_csv.iloc[0:num_rows]
        y_pred = self.model.predict( df_csv_slice )
        logging.info( ' ypred = ' + str(y_pred))

        # Compute RMSE.
        y_test = df_csv_slice['outcome_total_delivery_time']
        RMSE = np.sqrt( mean_squared_error(y_test, y_pred) )
        logger.info( 'RMSE = ' + str(RMSE) )

        self.assertTrue( RMSE < 1500.0 )


def main():
    # Configure logging.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(process)d/%(threadName)s - %(name)s - %(levelname)s - %(message)s',
                        #stream=sys.stdout)
                        filename='./test_model.log',
                        filemode='w')
    logger = logging.getLogger('main()')

    unittest.main()



if __name__ == "__main__":
    main()

