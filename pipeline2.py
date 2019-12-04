import pandas, json, nltk, pickle, re, numpy, sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from bow import BOW
from feature_extractor import FeatureExtractor
from skl_nb import NBModel
from skl_log import LogReg


from sklearn.metrics import confusion_matrix

cached_stopwords = stopwords.words('english')



def preprocess(text):

    #Lower case and strip whitespace
    text = text.lower().strip()

    #Strip punctuation
    text = re.sub(r'[.,\-?!\'\"\$\%\#():]', '', text)

    #Tokenize text
    tokens = word_tokenize(text)

    #Remove stop words
    tokens = [token for token in tokens if token not in cached_stopwords]

    return tokens

if __name__ == '__main__':

    # train_file = "data_train.json"
    # all_avail = pandas.read_json(open(train_file, 'r'))
    
    # all_avail['text'] = all_avail['text'].apply(func=preprocess)
    # pickle.dump(all_avail, open('all_avail.pickle', 'wb'))
    all_avail = pickle.load(open('all_avail.pickle', 'rb'))
    print('Finished preprocessing')

    #80-20 Split for training and development
    train_set = all_avail.sample(frac=0.8, random_state=1)
    dev_set = all_avail.drop(train_set.index)

    #Instantiate bag-of-words object
    bow = BOW(train_set)

    #Create vocabulary
    bow.create_bigram_vocabulary(500)
    bow.create_vocabulary(500)
    print('Finished BOW')

    feats = FeatureExtractor(bow)
    train_X, train_label = feats.df_to_feats_skl(train_set)
    dev_X, dev_label = feats.df_to_feats_skl(dev_set)

    print(train_X)
    print(train_label)
    print('Finished converting instances to vectors')

    # #Train naive bayes and get its accuracy
    # nb = NBModel(train_X, train_label, dev_X, dev_label)
    # nb.train()
    # print("Naive Bayes Model Accurracy:")
    # print(nb.validate())

    #Train logistic regression and get its accurracy
    log_reg = LogReg(train_X, train_label, dev_X, dev_label)
    log_reg.train()
    print("\nLogistic Regression Model Accurracy: {}".format(log_reg.validate()))






