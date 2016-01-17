import os
import glob
from operator import itemgetter

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer


class SongRecommender(object):
    """

    """

    def __init__(self, data_path=None):
        """

        :param data_path:
        :return:
        """
        self.data = {}
        self.path = data_path
        self.tfidf = TfidfTransformer()
        self.song_vectors = None
        self.user_profiles = None
        self.song_names = []
        self.song_features = []



    def transform_data(self):

        self.song_vectors = self._song_tfidf()
        self.user_profiles = self._user_tastes()

    def recommend(self, user, n_songs=3):
        """

        :param user:
        :param n_songs:
        :return:
        """

        predicted_tastes = self.user_profiles[user].multiply(self.song_vectors)
        # get mask of already listened
        already_listend = self.data["user"].iloc[user, self.data["user"] != 0]
        self.masked_preds = predicted_tastes.iloc[already_listend] = 0
        zip(song_names,self.masked_preds)
        return self.song_names[]



    def _prep_data(self):
        """

        """
        self._check_data_path()
        self._load_data()
        self.transform_data()

    def _user_tastes(self):
        """

        :return:
        """

        user_prefs = self.data["user"]

        for i, user in enumerate(user_prefs.columns()):
            user_profiles = np.zeros(len(user_prefs), len(self.data["song"].columns))
            user_profiles[i, :] = self.song_vectors.transpose().multiply(user_prefs[user], fill_value=0)
            self.user_profiles = pd.DataFrame(data=user_profiles, index=self.data["user"].columns,
                                              columns=self.song_features)

    def _song_tfidf(self):
        """

        :return:
        """
        self.songs = list(self.data["song"].index)
        self.song_features = list(self.data["song"].columns)
        return self.tfidf.transform(self.data["song"])

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
        Loads csv into DataFrames

        """
        try:
            # strip trailing / if exists
            path = self.path[:-1] if self.path[-1] == "/" else self.path
            data_files = glob.glob("{0}/*.csv".format(path))
            for csv in data_files:
                # with current files results in engagement and users
                data_key = csv.split("_")[0]
                self.data[data_key] = pd.DataFrame.from_csv(csv).fillna(0)
                print("loaded %s" % csv)

        except IOError as e:
            print("error occurred loading csv files: {0}".format(e.message))
