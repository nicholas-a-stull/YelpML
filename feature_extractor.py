import collections
import json
import pprint
from pandas import Series


from bow import BOW

class FeatureExtractor():
    def __init__(self, bow):
        self.bow = bow


    def df_to_feats(self, df):

        df['text'] = df['text'].apply(self.extract_all)
        #df['stars'] = df['stars'].apply(lambda x : 'pos' if x >= 3 else 'neg')

        feature_series = df.loc[:, ['text', 'stars']]

        return list(feature_series.itertuples(index=False, name=None))

    def get_negative_words(self, tokens):
        negative_words = {"terrible", "horrible", "waste", "never", "instead", "disappointed", "cold", "guess", "girl"}
        num_negative = 0
        for token in tokens:
            if token in negative_words:
                num_negative += 1
        return num_negative

    def get_positive_words(self, tokens):
        positive_words = {"perfectly", "back!", "great!", "reviews", "perfect", "care", "amazing", "friendly"}
        num_positive = 0
        for token in tokens:
            if token in positive_words:
                num_positive += 1
        return num_positive

    def extract_all(self, tokens):

        features = {}

        features.update(self.bow.get_vocabulary_features(tokens))
        features.update(self.bow.get_bigram_features(tokens))
        # features['num_pos'] = self.get_positive_words(tokens)
        # features['num_neg'] = self.get_negative_words(tokens)

        return features

