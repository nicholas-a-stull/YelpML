# YelpML

###Description of Files:

    - bow.py: Bag-of-Words object to create a unigram and bigram vocabulary, and used to generate
        bow features.
    - pipeline.py: Runs the whole pipeline for classification, including preprocessing, creating vocabulary,
       converting text features to vectors and running the model.
    - feature_extractor.py: Handles feature extraction for dataframes
    - skl_{nb,log,ptron}.py: Simple files to handle training and validation for their respective models
    
    
###Notes on running and runtime:
    - Features are saved based on last saved model -- may change depending on how much of the dataset was
       used to train. predictions.csv was generated using feats generated from ALL of the train dataset.
    - Loading the data and preprocessing the data takes a decent amount of time due to tokenization
    - To save time, the pipeline pickles the preprocessed data, so if you are running the code twice,
       uncomment/comment the correct lines to only run preprocessing once.
    - Model selection is handled at the end of the pipeline, using one of the model objects.
    - Prediction can be done on any trained or loaded model
    - Naive Bayes may cause a Segmentation Fault, this happened only when other programs were running at the same time
    - log_reg.pickle and nb.pickle are pretrained models, which can be loaded as specified below. 
    

###How to run the system

#### Requirements
    - pip requirements found in requirements.txt
    - nltk requires: punkt, stopwords

####From Scratch (note: to run preprocessing, uncomment out the block in load_data of pipeline.py, lines 41-43)
    - Import correct files (example in main() of pipeline.py
    - Create a pipeline object : pipe = Pipeline()
    - Train the model with arguments: pipe.run_training(train_file, validate, run_nb, run_lr)
        - train_file : file to train on
        - validate   : whether or not to do 80-20 split of training data and return validation accuracy
        - run_nb     : True/False, whether or not to train NaiveBayes model (default: False)
        - run_lr     : True/False, whether or not to train Logistic Regression model (default: False)
    - Get predictions: pipe.predict(test_file, model_type)
        - model_type : either 'lr', 'nb', 'pt'
    - Print predictions to csv: pipe.pred_to_csv(prediction)
    
    
####Using Pre-trained model
    - Make sure necessary files are in pickles folder (feats.pickle, (log_reg/nb).pickle)
    - Create pipeline: pipe = Pipeline()
    - Load model: pipe.load_model(model_type)
        -model_type : either 'lr' or 'nb', 'pt'
    - Load feature generator: pipe.feats = joblib.load('feats.pickle')
    
    Getting and printing predictions are same as above
    
    
    
    
    
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