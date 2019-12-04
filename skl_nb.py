import nltk
from sklearn.naive_bayes import GaussianNB


class NBModel():

    def __init__(self, train_x, train_y, dev_x, dev_y):

        self.train_x = train_x
        self.train_y = train_y
        self.dev_x = dev_x
        self.dev_y = dev_y


        self.model = GaussianNB()


    def train(self):
        self.model.fit(self.train_x, self.train_y)

    def validate(self):
         return self.model.score(self.dev_x, self.dev_y)

    def predict(self, test):
        return self.model.predict(test)




