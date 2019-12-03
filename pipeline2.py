import pandas, json, nltk, pickle, re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from bow import BOW
from feature_extractor import FeatureExtractor
from nltk_nb import NBModel

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

    train_file = "data_train.json"
    all_avail = pandas.read_json(open(train_file, 'r'))[:50000]

    all_avail['text'] = all_avail['text'].apply(func=preprocess)
    print('Finished preprocessing')

    #80-20 Split for training and development
    train_set = all_avail.sample(frac=0.8, random_state=1)
    dev_set = all_avail.drop(train_set.index)

    #Instantiate bag-of-words object
    bow = BOW(train_set)

    #Create vocabulary
    #bow.create_vocabulary(n=500)
    bow.create_bigram_vocabulary(500)
    bow.create_vocabulary(500)
    print('Finished BOW')

    feats = FeatureExtractor(bow)
    train_feats = feats.df_to_feats(train_set)
    dev_feats = feats.df_to_feats(dev_set)


    model = NBModel(train_feats, dev_feats)
    print('Finished Training')
    pickle.dump(model, open('model.pickle', 'wb'))

    print(model.validate())






