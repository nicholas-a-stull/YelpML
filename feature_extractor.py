import collections
import json
import pprint
from pandas import Series
import numpy


from bow import BOW

class FeatureExtractor():
    def __init__(self, bow):
        self.bow = bow


    def df_to_feats_nltk(self, df):

        df['text'] = df['text'].apply(self.extract_all)
        #df['stars'] = df['stars'].apply(lambda x : 'pos' if x >= 3 else 'neg')

        feature_series = df.loc[:, ['text', 'stars']]

        return list(feature_series.itertuples(index=False, name=None))

    def df_to_feats_skl(self, df, label=True):

        #Extract features from tokens
        df['text'] = df['text'].apply(self.extract_all)

        X = numpy.vstack(df.loc[:, 'text'].to_numpy())

        label_ret = None
        if label:
            label_ret = df.loc[:, 'stars'].to_numpy()

        return X, label_ret


    def extract_all(self, tokens):

        features = numpy.concatenate((self.bow.get_vocabulary_features(tokens), self.bow.get_bigram_features(tokens)))

        return features

