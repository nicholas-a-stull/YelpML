import collections
import json
import pprint


def extract_all(instance):
    features = {}

    features["length"] = get_length(instance)
    features["1-star-words"] = get_one_star_words(instance)
    features["useful"] = instance["useful"]
    features["funny"] = instance["funny"]
    features["cool"] = instance["cool"]


def get_length(instance):
    return len(instance["tokens"])


def get_one_star_words(instance):
    one_star_words = {"horrible"}
    text_as_set = set(instance["tokens"])
    res = text_as_set.intersection(one_star_words)
    return len(res)



#     return
#
# def 2_star_words(instance):
#
# def 3_star_words(instance):
#
# def 4_star_words(instance):
#
# def 5_star_words(intance):


if __name__ == "__main__":
    test_instance = {
        "text": ["Hello", ",", "world", "!"],
        "useful": 0,
        "funny": 1,
        "cool": 1
    }

    with open("data_train.json") as train_data:
        data = json.load(train_data)
        result = {
            "1-star": collections.Counter(),
            "2-star": collections.Counter(),
            "3-star": collections.Counter(),
            "4-star": collections.Counter(),
            "5-star": collections.Counter()
        }


        for i in data:
            # tokens = i['text'].split()
            # bigrams = [(tokens[j].lower(), tokens[j+1].lower()) for j in range(0,len(tokens)-1)]
            # print(bigrams)
            if i["stars"] == 1:
                result["1-star"].update(i["text"].split())
            elif i["stars"] == 2:
                result["2-star"].update(i["text"].split())
            elif i["stars"] == 3:
                result["3-star"].update(i["text"].split())
            elif i["stars"] == 4:
                result["4-star"].update(i["text"].split())
            elif i["stars"] == 5:
                result["5-star"].update(i["text"].split())

        pprint.pprint(result["2-star"].most_common(500))





    # result = extract_all(test_instance)
    #
    # print (result)
