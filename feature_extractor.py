import collections
import json
import pprint
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sent = SentimentIntensityAnalyzer()

def extract_all(instance):
    features = {}
    
    features["length"] = get_length(instance)
    features["negative-words"] = get_negative_words(instance)
    features["positive-words"] = get_positive_words(instance)
    features["more-positive-words"] = get_more_positive_or_negative(features)
    features["most-common-number"] = get_most_common_number(instance)
    features["most-common-stars"] = get_most_common_stars(instance)
    features["contains-smiley-face"] = get_contains_smiley_face(instance)
    
    features['vader_sentiment'] = get_vader_sentiment(' '.join(instance['tokens']))
    
    features["useful"] = instance["useful"]
    features["funny"] = instance["funny"]
    features["cool"] = instance["cool"]

    return features

def get_vader_sentiment(instance):
    return sent.polarity_scores(instance)['compound']

def get_length(instance):
    return len(instance["tokens"])


def get_negative_words(instance):
    negative_words = {"terrible", "horrible", "waste", "never", "instead", "disappointed", "cold", "guess", "girl"}
    num_negative = 0
    for token in instance["tokens"]:
        if token in negative_words:
            num_negative += 1
    return num_negative


def get_positive_words(instance):
    positive_words = {"perfectly", "back!", "great!", "reviews", "perfect", "care", "amazing", "friendly"}
    num_positive = 0
    for token in instance["tokens"]:
        if token in positive_words:
            num_positive += 1
    return num_positive


def get_more_positive_or_negative(features):
    if features["positive-words"] > features["negative-words"]:
        return 1
    else:
        return 0


def get_most_common_number(instance):
    num_ones = 0
    num_twos = 0
    num_threes = 0
    num_fours = 0
    num_fives = 0
    for token in instance["tokens"]:
        if token == "1":
            num_ones += 1
        elif token == "2":
            num_twos += 1
        elif token == "3":
            num_threes += 1
        elif token == "4":
            num_fours += 1
        elif token == "5":
            num_fives += 1
    highest_val = max(num_ones, num_twos, num_threes, num_fours, num_fives)
    if highest_val == num_ones:
        return 1
    elif highest_val == num_twos:
        return 2
    elif highest_val == num_threes:
        return 3
    elif highest_val == num_fours:
        return 4
    elif highest_val == num_fives:
        return 5


def get_most_common_stars(instance):
    num_ones = 0
    num_twos = 0
    num_threes = 0
    num_fours = 0
    num_fives = 0
    for i in range(0, len(instance["tokens"]) - 1):
        if instance["tokens"][i] == "1" and (instance["tokens"][i+1] == "stars" or instance["tokens"][i+1] == "star"):
            num_ones += 1
        elif instance["tokens"][i] == "2" and (instance["tokens"][i + 1] == "stars" or instance["tokens"][i+1] == "star"):
            num_twos += 1
        elif instance["tokens"][i] == "3" and (instance["tokens"][i + 1] == "stars" or instance["tokens"][i+1] == "star"):
            num_threes += 1
        elif instance["tokens"][i] == "4" and (instance["tokens"][i + 1] == "stars" or instance["tokens"][i+1] == "star"):
            num_fours += 1
        elif instance["tokens"][i] == "5" and (instance["tokens"][i + 1] == "stars" or instance["tokens"][i+1] == "star"):
            num_fives += 1
    highest_val = max(num_ones, num_twos, num_threes, num_fours, num_fives)
    if highest_val == num_ones:
        return 1
    elif highest_val == num_twos:
        return 2
    elif highest_val == num_threes:
        return 3
    elif highest_val == num_fours:
        return 4
    elif highest_val == num_fives:
        return 5


def get_contains_smiley_face(instance):
    for i in instance["tokens"]:
        if i == ":)":
            return 1
    return 0


if __name__ == "__main__":
    test_instance = {
        "tokens": ["horrible", "horrible", "world", "!", "2", "star", "2", "stars"],
        "useful": 0,
        "funny": 1,
        "cool": 1
    }

    pprint.pprint(extract_all(test_instance))
