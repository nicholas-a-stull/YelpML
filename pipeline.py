import pandas, json, nltk, pickle, re, numpy, sklearn, joblib
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

    def load_data(self, file, validate=True, label=True):

        fname = file[:-5]

        # all_avail = pandas.read_json(open(file, 'r'))
        # all_avail['text'] = all_avail['text'].apply(func=self.preprocess)
        # pickle.dump(all_avail, open('pickles/'+fname + '.pickle', 'wb'))

        all_avail = pickle.load(open('pickles/'+fname + '.pickle', 'rb'))
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

        self.feats = FeatureExtractor(bow)
        train_X, train_label = self.feats.df_to_feats_skl(train_set, label)

        joblib.dump(self.feats, 'pickles/feats.pickle')
        print('dumped nb')
        if validate:
            dev_X, dev_label = self.feats.df_to_feats_skl(dev_set, label)
            return train_X, train_label, dev_X, dev_label

        else:
            return train_X, train_label

    def run_training(self, input_file, validate=True, label=True, run_lr = False, run_nb = False):

        if validate:
            train_X, train_label, dev_X, dev_label = self.load_data(input_file, validate, label)
        else:
            train_X, train_label = self.load_data(input_file, validate, label)
            dev_X = None
            dev_label = None

        print('Finished converting instances to vectors')

        #Train naive bayes and get its accuracy
        if run_nb:
            nb = NBModel(train_X, train_label, dev_X, dev_label)
            nb.train()
            self.nb = nb
            if validate:
                print("Naive Bayes Model Accurracy:")
                print(nb.validate())
            joblib.dump(nb, 'pickles/nb.pickle')

        # Train logistic regression and get its accurracy
        if run_lr:
            log_reg = LogReg(train_X, train_label, dev_X, dev_label)
            log_reg.train()
            self.log_reg = log_reg
            if validate:
                print("\nLogistic Regression Model Accurracy: {}".format(log_reg.validate()))
            joblib.dump(log_reg, 'pickles/log_reg.pickle')


    def predict_file(self, test_file, model_type):

        #Load and preprocess the test set
        test = pandas.read_json(open(test_file, 'r'))
        test['text'] = test['text'].apply(func=self.preprocess)
        print('Finished preprocessing test file')

        #Do not create new vocabulary, use one from training
        test_x = self.feats.df_to_feats_skl(test, label=False)
        test_x = test_x[0]
        print('Finished converting test instances to vectors')

        pred = None

        if model_type == 'nb' and self.nb:
            pred = self.nb.predict(test_x)

        elif model_type == 'lr' and self.log_reg:
            pred = self.log_reg.predict(test_x)

        return pred

    def load_model(self, model_type):
        if model_type == 'nb':
            self.nb = joblib.load('pickles/nb.pickle')

        elif model_type == 'lr':
            self.log_reg = joblib.load('pickles/log_reg.pickle')

        else:
            print('No model loaded.')

    def pred_to_csv(self, pred):
        df = pandas.DataFrame({'Predictions' : pred}, dtype='float')
        df.to_csv('predictions.csv', index=False)




def main():

    #This block runs the whole pipeline, including training (excludes preprocessing)
    # pipe = Pipeline()
    # pipe.run_training('data_train.json', validate=False, run_lr=False, run_nb=True)
    # print('Finished Training')
    # predictions = pipe.predict_file('data_test_wo_label.json', model_type= 'nb')
    # print(predictions)

    #This block loads models from the pickle directory
    pipe = Pipeline()
    pipe.load_model('nb')
    pipe.feats = joblib.load('pickles/feats.pickle')
    predictions = pipe.predict_file('data_test_wo_label.json', model_type = 'nb')
    #pipe.pred_to_csv(predictions)
    print(predictions)

if __name__ == '__main__':
    main()


