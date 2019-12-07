# YelpML

## By: Abteen Ebrahimi, Michael Hypnarowski, Nicholas Stull

## 12 December 2019, CSE 142 (UCSC)

### Files:

- pipeline.py: Runs the whole pipeline for classification, including preprocessing, creating vocabulary,
converting text features to vectors and running the model.

- bow.py: Bag-of-Words object to create a unigram and bigram vocabulary. Also used to generate
bow features.

- feature_extractor.py: Handles feature extraction for dataframes.

- skl_{nb,log,ptron}.py: Simple files to handle training and validation for their respective models.
    
    
### Runtime and Related Notes:
- The run time is about 2 minutes and 10 seconds for a full run with pre-processing, training, and validation.

- Features are saved based on last saved model, which may change depending on how much of the dataset was
   used to train. The predictions csv was generated using features generated from ALL of the train dataset.
   
- Loading the data and preprocessing the data takes a long time due to tokenization.

- To save time, the pipeline pickles the preprocessed data, so if you are running the code twice,
   uncomment/comment the correct lines designated in pipeline.py to only run preprocessing once.
   
- Model selection is handled at the end of the pipeline, using one of the model objects.

- Prediction can be done on any trained or loaded model.

- Naive Bayes may cause a Segmentation Fault, this happened only when other programs were running at the same time.

- log_reg.pickle and nb.pickle are pretrained models, which can be loaded as specified below. 
    

### Running the System:

#### Requirements:

- See requirements.txt for information about required libraries.

- nltk requires: punkt, stopwords

#### Running From Scratch (note: to run preprocessing, uncomment out the block in load_data of pipeline.py, lines 41-43):

- Create a pipeline object: pipe = Pipeline()

- Train the model with arguments: pipe.run_training(train_file, validate, run_nb, run_lr)
    - train_file : file to train on
    - validate   : whether or not to do 80-20 split of training data and return validation accuracy
    - run_nb     : True/False, whether or not to train NaiveBayes model (default: False)
    - run_lr     : True/False, whether or not to train Logistic Regression model (default: False)

- Get predictions: pipe.predict(test_file, model_type). model_type: either 'lr', 'nb', 'pt'.

- Print predictions to csv: pipe.pred_to_csv(prediction)
    
    
#### Using the Pre-trained model

- Make sure necessary files are in the pickles folder (feats.pickle, (log_reg/nb).pickle)

- Create pipeline: pipe = Pipeline()

- Load model: pipe.load_model(model_type). model_type : either 'lr' or 'nb', 'pt'.

- Load feature generator: pipe.feats = joblib.load('feats.pickle')

- Get predictions: pipe.predict(test_file, model_type). model_type: either 'lr', 'nb', 'pt'.

- Print predictions to csv: pipe.pred_to_csv(prediction)
