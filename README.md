# YelpML

###Description of Files:

    - bow.py: Bag-of-Words object to create a unigram and bigram vocabulary, and used to generate
        bow features.
    - pipeline.py: Runs the whole pipeline for classification, including preprocessing, creating vocabulary,
       converting text features to vectors and running the model.
    - feature_extractor.py: Handles feature extraction for dataframes
    - skl_{nb,log}.py: Simple files to handle training and validation for their respective models
    
    
###Notes on running and runtime:
    - Loading the data and preprocessing the data takes a decent amount of time due to tokenization
    - To save time, the pipeline pickles the preprocessed data, so if you are running the code twice,
       uncomment/comment the correct lines to only run preprocessing once.
    - Model selection is handled at the end of the pipeline, using one of the model objects.
    
    
    
    
    
Notes for writeup:

Preprocessing:
    While the data is structured, with information such as the cool, useful, and funny counts, we found that these
    attributes were not helpful features in our models. As the only usable feature was the text field, we decided to preprocess 
    the data in order to reduce noise present in the reviews. To do this, we lowercase all the text, strip whitespace and 
    punctuation, as well as tokenize the text using NLTK. We then use NLTK's built-in corpora of stop words to remove anything
    that has a low chance of being informative. Due to this preprocessing, the text attribute of each instance is converted 
    to a bag of words representation covering the whole instance, and we lose sentence structure. 
    
    The next step in our preprocessing is creating a vocabulary of unigrams and bigrams generated from our dataset. 
    We keep the 500 most frequent from each set, and create an indexed vocabulary, i.e. a dictionary mapping each token to a unique integer. 
    The value of 500 is arbitary, and selected through trial and error.
    This vocabulary is used in feature generation. Although we originally wanted to create a vocabulary conditioned on instance labels, we found that this method yielded better results.
    
Feature Creation:
    To convert the bag of words for each instance into a structured vector, we use the generated dictionaries. Respectively for both unigrams and bigrams,
    we create a 500 dimensional vector, which represents the counts present in the given instance.
    Therefore, v_i = count(token) where vocabulary(token) = i.
    The process for bigram features is identical. 
    
Model Usage:
    Since our feature vectors and labels are all numpy arrays, we can easily use the models provided by scikit-learn. 
    We found that these implementations had reasonable running time and were easy to experiment with. 
    
    Logistic Regression: Hyperparamter selction