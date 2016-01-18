import os
import glob
from operator import itemgetter
import pdb

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
        self._prep_data()

    def transform_data(self):
        self.song_vectors = self._song_tfidf().todense()
        self.user_profiles = self._user_tastes()
        print("transformed song and user preference data")

    def recommend(self, user, n_songs=3):
        """

        :param user:
        :param n_songs:
        :return:
        """
        predicted_tastes = np.squeeze(np.asarray(self.song_vectors.dot(self.user_profiles.loc[user].values)))
        # get mask of already listened
        already_listend = self.data["user"][user].loc[self.data["user"][user] != 0]
        listened_index = [self.song_names.index(song) for song in already_listend.index]
        np.put(predicted_tastes,listened_index,-10)
        name_preds = zip(self.song_names, predicted_tastes)
        s_scores = [s_score for s_score in name_preds]
        s_scores.sort(key=itemgetter(1),reverse=True)
        return s_scores[:n_songs]

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
        user_profiles = np.zeros((len(user_prefs.columns), len(self.data["song"].columns)))

        for i, user in enumerate(user_prefs.columns):
            user_profiles[i, :] = self.song_vectors.transpose().dot(user_prefs[user])

        prefs_df= pd.DataFrame(data=user_profiles, index=self.data["user"].columns, columns=self.song_features)
        return prefs_df


    def _song_tfidf(self):
        """

        :return:
        """
        self.song_names = list(self.data["song"].index)
        self.song_features = list(self.data["song"].columns)
        self.tfidf.fit(self.data["song"])
        return self.tfidf.transform(self.data["song"])

    def _check_data_path(self):
        """
        Checks if a data path was provided otherwise defaults to data contained in package
        """
        if not self.path:
            dirname, fname = os.path.split(os.path.abspath(__file__))
            self.path = dirname + "/data"

        print("using data path: {0}".format(self.path))

    def _load_data(self):
        """
        Loads csv into DataFrames

        """
        try:
            path = self.path
            data_files = glob.glob("{0}/*.csv".format(self.path))
            for csv in data_files:
                # with current files results in engagement and users
                data_key = csv.split(".csv")[0].split("/")[-1].split("_")[0]
                self.data[data_key] = pd.DataFrame.from_csv(csv).fillna(0)
                print("loaded %s" % csv)

        except IOError as e:
            print("error occurred loading csv files: {0}".format(e.message))
