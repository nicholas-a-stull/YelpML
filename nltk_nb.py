import nltk



class NBModel():

    def __init__(self, train_set, dev_set):

        self.train_set = train_set
        self.dev_set = dev_set

        self.nb = nltk.NaiveBayesClassifier.train(self.train_set)


    def validate(self):
        return nltk.classify.accuracy(self.nb, self.dev_set)

    def informative_features(self, n=3):
        return self.nb.show_most_informative_features(n)




