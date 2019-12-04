from sklearn.linear_model import LogisticRegression

class LogReg():

    def __init__(self, tx, ty, dx, dy):

        self.train_x = tx
        self.train_y = ty
        self.dev_x = dx
        self.dev_y = dy

        self.model = LogisticRegression()

    def train(self):
        self.model.fit(self.train_x, self.train_y)

    def validate(self):
        return self.model.score(self.dev_x, self.dev_y)

    def predict(self, test):
        return self.model.predict(test)

