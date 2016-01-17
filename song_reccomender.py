import os
import glob

import pandas as pd
import numpy as np


class SongReccomender(object)
    def __init__(self, data_path = None):
        self.data = None
        self.path = data_path

    def user_correlation(self,user1,user2):
        pass

    def _check_data_path(self):
            """
            Checks if a data path was provided otherwise defaults to data contained in package
            """
            if not self.path:
                dirname, fname = os.path.split(os.path.abspath(__file__))
                self.path = '/'.join(dirname.split('/')[:-1]) + "/data/"

            print("using data path: {0}".format(self.path))

    def _load_data(self):
        """
        Loads xls  into DataFrame

        """
        try:

            # strip trailing / if exists
            path = self.path[:-1] if self.path[-1] == "/" else self.path
            data_file = glob.glob("{0}/*.xls".format(path))
            self.data = pd.DataFrame.read_excel(data_file)
            print("loaded %s" % data_file)

        except IOError as e:
            print("error occurred loading xls data: {0}".format(e.message))



