import pandas, json, nltk, pickle, re, numpy, sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from bow import BOW
from feature_extractor import FeatureExtractor
from skl_nb import NBModel
from skl_log import LogReg


from sklearn.metrics import confusion_matrix



class Pipeline():

    def __init__(self):
        self.cached_stopwords = stopwords.words('english')
        self.nb = None
        self.log_reg = None

    def preprocess(self,text):
        # Lower case and strip whitespace
        text = text.lower().strip()

        # Strip punctuation
        text = re.sub(r'[.,\-?!\'\"\$\%\#():]', '', text)

        # Tokenize text
        tokens = word_tokenize(text)

        # Remove stop words
        tokens = [token for token in tokens if token not in self.cached_stopwords]

        return tokens

    def load_data(self, file, validate=True):

        fname = file[:-5]

        # all_avail = pandas.read_json(open(file, 'r'))
        # all_avail['text'] = all_avail['text'].apply(func=self.preprocess)
        # pickle.dump(all_avail, open(fname + '.pickle', 'wb'))



        all_avail = pickle.load(open(fname + '.pickle', 'rb'))
        print('Finished preprocessing')

        # 80-20 Split for training and development
        if validate:
            train_set = all_avail.sample(frac=0.8, random_state=1)
            dev_set = all_avail.drop(train_set.index)
        else:
            train_set = all_avail

        # Instantiate bag-of-words object
        bow = BOW(train_set)

        # Create vocabulary
        bow.create_bigram_vocabulary(500)
        bow.create_vocabulary(500)
        print('Finished BOW')

        feats = FeatureExtractor(bow)
        train_X, train_label = feats.df_to_feats_skl(train_set)
        if validate:
            dev_X, dev_label = feats.df_to_feats_skl(dev_set)
            return train_X, train_label, dev_X, dev_label

        else:
            return train_X, train_label

    def run_training(self, input_file, validate=True):

        if validate:
            train_X, train_label, dev_X, dev_label = self.load_data(input_file, validate)
        else:
            train_X, train_label = self.load_data(input_file, validate)
            dev_X = None
            dev_label = None

        print('Finished converting instances to vectors')

        #Train naive bayes and get its accuracy
        # nb = NBModel(train_X, train_label, dev_X, dev_label)
        # nb.train()
        # self.nb = nb
        # if validate:
        #     print("Naive Bayes Model Accurracy:")
        #     print(nb.validate())

        # Train logistic regression and get its accurracy
        log_reg = LogReg(train_X, train_label, dev_X, dev_label)
        log_reg.train()
        self.log_reg = log_reg
        if validate:
            print("\nLogistic Regression Model Accurracy: {}".format(log_reg.validate()))

    def predict_file(self, test_file):

        test_x, test_y = self.load_data(self, test_file, validate=False)

        if self.nb:
            raise NotImplementedError

        if self.log_reg:
            raise NotImplementedError






def main():
    pipe = Pipeline()
    pipe.run_training('data_train.json', validate=True)

if __name__ == '__main__':
    main()


