import pandas, json, nltk, pickle
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords

from bow import BOW
from feature_extractor import FeatureExtractor
from nltk_nb import NBModel

cached_stopwords = stopwords.words('english')
tokenizer = ToktokTokenizer()



def nltk_preprocess(text, remove_stopwords = True):
    tokens = tokenizer.tokenize(text)

    if remove_stopwords:
        tokens = [token.lower() for token in tokens if token not in cached_stopwords]

    return tokens

if __name__ == '__main__':

    train_file = "data_train.json"
    all_avail = pandas.read_json(open(train_file, 'r'))[:1000]

    all_avail['text'] = all_avail['text'].apply(func=nltk_preprocess)
    print('Finished preprocessing')

    train_set = all_avail.sample(frac=0.8, random_state=1)
    dev_set = all_avail.drop(train_set.index)

    #Calculate set of words independent to each label
    bow = BOW(train_set)
    #bow.calculate_independent_words()
    bow.create_vocabulary()

    print('Finished BOW')

    feats = FeatureExtractor(bow)
    train_feats = feats.df_to_feats(train_set)
    dev_feats = feats.df_to_feats(dev_set)

    #print(train_feats)

    model = NBModel(train_feats, dev_feats)
    print('Finished Training')
    pickle.dump(model, open('model.pickle', 'wb'))

    print(model.validate())






