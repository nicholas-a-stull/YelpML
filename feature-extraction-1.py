import collections
import json
import pprint


def extract_all(instance):
    features = {}

    features["length"] = get_length(instance)
    features["negative-words"] = get_negative_words(instance)
    features["useful"] = instance["useful"]
    features["funny"] = instance["funny"]
    features["cool"] = instance["cool"]

    return features


def get_length(instance):
    return len(instance["tokens"])


def get_negative_words(instance):
    negative_words = {"horrible"}
    num_negative = 0
    for token in instance["tokens"]:
        if token in negative_words:
            num_negative += 1
    return num_negative



if __name__ == "__main__":
    test_instance = {
        "tokens": ["horrible", "horrible", "world", "!"],
        "useful": 0,
        "funny": 1,
        "cool": 1
    }

    result = extract_all(test_instance)

    print(result)

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
                
            pprint.pprint(result["3-star"])
    
