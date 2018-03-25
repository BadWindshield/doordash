from datetime import datetime
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
import sys
import time
import unittest


class TestModel(unittest.TestCase):
    """
    Apply the model to a few training samples.
    """
    def setUp(self):
        logger = logging.getLogger('TestModel.setUp()')

    def tearDown(self):
        logger = logging.getLogger('TestModel.tearDown()')

    def test_predictions(self):
        logger = logging.getLogger('TestModel.test_predictions()')


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

