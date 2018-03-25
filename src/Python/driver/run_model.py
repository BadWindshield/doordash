from datetime import datetime
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
import sys
import time


def main():
    # Configure logging.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(process)d/%(threadName)s - %(name)s - %(levelname)s - %(message)s',
                        #stream=sys.stdout)
                        filename='./run_model.log',
                        filemode='w')
    logger = logging.getLogger('main()')



if __name__ == "__main__":
    main()

