from collections import defaultdict, Counter
import nltk, numpy

class BOW():

    word_sets = [set(), set(), set(), set(), set()]

    def __init__(self, df):
        self.df = df


    def calculate_independent_words(self):

        for i in range(1,6):
            series = self.df.loc[self.df['stars'] == i]
            flat = [x for y in series['text'].values.tolist() for x in y]
            self.word_sets[i-1] = set(flat)

        self.intersection = set.intersection(*self.word_sets)

        for i in range(0,5):
            self.word_sets[i] = self.word_sets[i].difference(self.intersection)
            print(len(self.word_sets[i]))

    def get_by_star_bow(self, tokens):
        bow_dict = {
            '1Star_BOW': 0,
            '2Star_BOW': 0,
            '3Star_BOW': 0,
            '4Star_BOW': 0,
            '5Star_BOW': 0
        }
        for token in tokens:
            for i in range(0, 5):
                if token in self.word_sets[i]:
                    bow_dict[str(i + 1) + 'Star_BOW'] += 1

        return bow_dict

    def get_vocabulary_features(self, tokens):
        temp = numpy.zeros(self.vocabulary_n, dtype='int')
        for tok in tokens:
            try:
                temp[self.vocabulary_vector[tok]] += 1
            except:
                pass
        return temp

    def create_vocabulary(self, n=50):
        text_series = self.df['text']
        tokens = [x for instances in text_series.values.tolist() for x in instances]
        counted = Counter(tokens)

        self.vocabulary = [x[0] for x in counted.most_common(n)]
        self.vocabulary_vector = {}
        self.vocabulary_n = n
        for i, word in enumerate(self.vocabulary):
            self.vocabulary_vector[word] = i
        #self.vocabulary_vector['<unk>'] = 0

    def get_bigrams(self,li):
        return list(nltk.bigrams(li))

    def create_bigram_vocabulary(self, n=50):
        text_series = self.df['text']

        text_series_bigrams = text_series.apply(func=self.get_bigrams)

        all_bigrams = [bigram for instances in text_series_bigrams.values.tolist() for bigram in instances]
        counted_bigrams = Counter(all_bigrams)

        self.bigrams = [x[0] for x in counted_bigrams.most_common(n)]
        self.bigram_vector = {}
        self.bigram_n = n
        for i,bigram in enumerate(self.bigrams):
            self.bigram_vector[bigram] = i
        #self.bigram_vector['<unk>'] = 0

    def get_bigram_features(self, tokens):
        temp = numpy.zeros(self.bigram_n, dtype='int')
        bigrams = nltk.bigrams(tokens)

        for bigram in bigrams:
            try:
                temp[self.bigram_vector[bigram]] += 1
            except:
                pass
        return temp
