import json, pickle, os, math
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords

cached_stopwords = stopwords.words('english')
tokenizer = ToktokTokenizer()

def nltk_preprocess(text, remove_stopwords = True):
    tokens = tokenizer.tokenize(text)

    if remove_stopwords:
        tokens = [token for token in tokens if token not in cached_stopwords]

    return tokens

def preprocess(train_set):


    # iterate through trainset and preprocess data
    for id, instance in enumerate(train_set):
        if id % 100000 == 0:
            print(id)
        tokens = nltk_preprocess(instance['text'])
        train_set_preprocessed.append({
            'tokens': tokens,
            'useful': instance['useful'],
            'funny': instance['funny'],
            'cool': instance['cool']
        })

    #pickle.dump(train_set_preprocessed, open('train_set_stop.pickle', 'wb'))


if __name__ == '__main__':

    if not os.path.exists('train_set_stop.pickle'):
        train_file = "data_train.json"
        train_set = json.load(open(train_file, 'r'))
        print('Loaded Json')
        train_set_preprocessed = preprocess(train_set)
    else:
        train_set_preprocessed = pickle.load(open('train_set_stop.pickle', 'rb'))
        print('Loaded from Pickle')


    #80-20 split for train and dev
    train_percent = 80
    train_value = math.floor(.8*len(train_set_preprocessed))

    train_set = train_set_preprocessed[:train_value]
    dev_set = train_set_preprocessed[train_value:]

    



