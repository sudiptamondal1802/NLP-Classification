#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv                                  # csv reader
import numpy as np                          # for array operations
import nltk                                 # Natural language toolkit
from sklearn.svm import LinearSVC           # implementing supervised learning
from nltk.classify import SklearnClassifier # classifier functions
from sklearn.pipeline import Pipeline       # used for cross validation with various steps and paramters
from sklearn.metrics import accuracy_score  # evaluation of the classifier 
from sklearn.metrics import precision_recall_fscore_support # evaluation of the classifier     
from nltk.tokenize import word_tokenize     # seperates strings into words (along with punctuations getting seperated)
from random import shuffle
nltk.download('punkt')


# In[2]:


# load data from a file and append it to the rawData
def loadData(path, Text=None):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            if line[0] == "DOC_ID":  # skip the header
                continue
            (Id, Text, Label) = parseReview(line)
            rawData.append((Id, Text, Label))


# In[3]:


def splitData(percentage):
    # A method to split the data between trainData and testData 
    dataSamples = len(rawData)
    halfOfData = int(len(rawData)/2)
    trainingSamples = int((percentage*dataSamples)/2)
    for (_, Text, Label) in rawData[:trainingSamples] + rawData[halfOfData:halfOfData+trainingSamples]:
        trainData.append((toFeatureVector(preProcess(Text)),Label))
    for (_, Text, Label) in rawData[trainingSamples:halfOfData] + rawData[halfOfData+trainingSamples:]:
        testData.append((toFeatureVector(preProcess(Text)),Label))


# # Question 1

# In[4]:


# Convert line from input file into an id/text/label tuple
def parseReview(reviewLine):
    # Should return a triple of an integer, a string containing the review, and a string indicating the label
    # Description:
    # As the file is tab separted, therefore using the csv reader the columns can be identified
    # from position 0 to 8 of the array structure
    # DOC_ID which is in the 0 position of the line has been used to assign the integer value
    # REVIEW_TEXT is in position 8 contains the review text 
    # LABEL is in position 1 and using conditions we can assign label1 as fake and label2 as real
    classname=""
    if reviewLine[1]=="__label1__":
        classname = "fake"
    else: 
        classname = "real"
    return (reviewLine[0], reviewLine[8], classname)


# In[5]:


# TEXT PREPROCESSING AND FEATURE VECTORIZATION

# Input: a string of one review
def preProcess(text):
    # Should return a list of tokens
    # Description:
    # word_tokenize of the tokenize package in the nltk library split sentences into words (also seperates punctuation)
    # including any punctuations of the text being passed as input parameter
    return word_tokenize(text)


# # Question 2

# In[6]:


featureDict = {} # A global dictionary of features

def toFeatureVector(tokens):
    # Should return a dictionary containing features as keys, and weights as values
    # Description:
    # In order to create the dictionary the tokens created by the preprocess step is 
    # used to execute the for loop. For every token 
    # validate if it is already present in the feature or local dictionary
    localDict = {}  
    for token in tokens:
        if token not in featureDict: 
            featureDict[token] = 0 
        else:
            featureDict[token] = 1  
   
        if token not in localDict:
            localDict[token] = 0   
        else:
            localDict[token] = 1   
    
    return localDict


# In[7]:


# TRAINING AND VALIDATING OUR CLASSIFIER
def trainClassifier(trainData):
    print("Training Classifier...")
    pipeline =  Pipeline([('svc', LinearSVC())])
    return SklearnClassifier(pipeline).train(trainData)


# # Question 3

# ### Description
# Cross validation evaluates a classifier performance by randomly dividing the training data into a development and training set as a first step, followed by computation of various scores to evaluate the performance of the classifier.
# The cross validation has been implemented below by executing a loop for 10 times or folds on the training set, the first 2 folds are shown as an example below:

# |Fold|Devset |Training set |
# |---------- |---------- |---------- |
# |1 |0-1679 |1680-16799 |
# |2 |1680-3359 |0-1679 & 3360-16799|

# Step 1: Train the classifer with the training set of the fold <br>
# Step 2: Predict labels for the devset based on the classifier samples received in the step 1 <br>
# Step 3: Compare the true labels and predicted labels for the devset to compute the precision, recall, f1-score and accuracy. <br>
# The higher the f1-score the better the classifier is, however the training in the test set should not be performed in order to avoid the overfitting.

# In[8]:


def crossValidate(dataset, folds):
    shuffle(dataset)
    cv_results = []
    foldSize = int(len(dataset)/folds) 
    #Description:
    #Loop in the logic to crossvalidate starting from start till end of record (16800th)
    #for every 1680 number of records each fold
    for i in range(0,len(dataset),foldSize):
        
        #1st fold contains : 1st 1680 records as the Devset followed by Training set
        #2nd fold contains : 1680 to 3360 as Devset and rest are training set and, so on
        
        trainingSet = dataset[:i]+dataset[foldSize+i:]
        classifier = trainClassifier(trainingSet)
        
        #predict label for the devset 
        devSet = dataset[i:i+foldSize]
        predLabel = predictLabels(devSet,classifier)
        
        #calcuate accuracy between true labels and predicted labels
        #retrieve the true label d[1] (fake or real) from the devset in order to compare with the predicted Label
        accuracy = accuracy_score(list(map(lambda d : d[1], devSet)), predLabel)
        
        #Calculate the precision, recall and F-score by comparing the true labels and predicted labels
        #retrieve the true label d[1] (fake or real) from the devset in order to compare with the predicted Label
        (precision,recall,fbeta_score,support) = precision_recall_fscore_support(list(map(lambda d : d[1], devSet)), predLabel, average ='weighted')
        
        #Populate the cv_results list with accuracy, precision, recall and F-score
        cv_results.append((accuracy,precision,recall,fbeta_score))
        
    #Convert the list in numpy array in order to be able to generate the mean of accuracy,precision,recall,fbeta_score respectively
    cv_results = (np.mean(np.array(cv_results),axis=0))
    return cv_results


# In[9]:


# PREDICTING LABELS GIVEN A CLASSIFIER

def predictLabels(reviewSamples, classifier):
    return classifier.classify_many(map(lambda t: t[0], reviewSamples))

def predictLabel(reviewSample, classifier):
    return classifier.classify(toFeatureVector(preProcess(reviewSample)))


# In[10]:


# MAIN

# loading reviews
# initialize global lists that will be appended to by the methods below
rawData = []          # the filtered data from the dataset file (should be 21000 samples)
trainData = []        # the pre-processed training data as a percentage of the total dataset (currently 80%, or 16800 samples)
testData = []         # the pre-processed test data as a percentage of the total dataset (currently 20%, or 4200 samples)

# the output classes
fakeLabel = 'fake'
realLabel = 'real'

# references to the data files
reviewPath = 'amazon_reviews.txt'

# Do the actual stuff (i.e. call the functions we've made)
# We parse the dataset and put it in a raw data list
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Preparing the dataset...",sep='\n')
loadData(reviewPath) 

# We split the raw dataset into a set of training data and a set of test data (80/20)
# You do the cross validation on the 80% (training data)
# We print the number of training samples and the number of features before the split
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Preparing training and test data...",sep='\n')
splitData(0.8)
# We print the number of training samples and the number of features after the split
print("After split, %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Training Samples: ", len(trainData), "Features: ", len(featureDict), sep='\n')

# QUESTION 3 - Make sure there is a function call here to the
# crossValidate function on the training set to get your results
print("Results from cross-validations (Accuracy, Precision, Recall, Fscore): ", crossValidate(trainData, 10))


# # Evaluate on test set

# In[11]:


# Finally, check the accuracy of your classifier by training on all the tranin data
# and testing on the test set
# Will only work once all functions are complete
functions_complete = True  # set to True once you're happy with your methods for cross val
if functions_complete:
    print(testData[0])   # have a look at the first test data instance
    classifier = trainClassifier(trainData)  # train the classifier
    testTrue = [t[1] for t in testData]   # get the ground-truth labels from the data
    testPred = predictLabels(testData, classifier)  # classify the test data to get predicted labels
    finalScores = precision_recall_fscore_support(testTrue, testPred, average='weighted') # evaluate
    print("Done training!")
    print("Precision: %f\nRecall: %f\nF Score:%f" % finalScores[:3])


# In[ ]:





# In[ ]:




