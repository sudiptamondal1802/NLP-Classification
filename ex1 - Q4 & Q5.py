#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv                                  # csv reader
import numpy as np                          # for array operations  
import nltk                                 # Natural language toolkit 
import string                               # string operations
from bs4 import BeautifulSoup               # python libray for extracting html data
from sklearn.svm import LinearSVC           # implementing supervised learning
from nltk.classify import SklearnClassifier # classifier functions
from sklearn.pipeline import Pipeline       # used for cross validation with various steps and paramters to form a pipeline of steps
from sklearn.metrics import accuracy_score  # evaluation of the classifier
from sklearn.metrics import precision_recall_fscore_support  # evaluation of the classifier
from nltk.corpus import stopwords           # for importing stopwords corpus
from nltk.stem import WordNetLemmatizer     # lemmatization
from sklearn.feature_extraction.text import TfidfVectorizer 
from random import shuffle
from nltk.stem import PorterStemmer
nltk.download('punkt')                      # sentence tokenizer
nltk.download('stopwords')                  # database of stopwords in english
nltk.download('wordnet')                    # provides a database of english words used fo lematization



# # Question 5

# ### Classifier improved when adding more features: 
# On adding more features corresponding to rating, verified purchase and product id, improves the accuracy, precision, recall and F-score of the classifier to about 20 percent along with the text processing improvements taken care in question 4.
# Feature selection is supposedly very important criteria in order to improve a classifier. Relevant data helps the discrimination into correct classes.
# For instance, when product category is selected as the feature the test data f-score is 78%, however using product id change the scores to 80% with 2% increase in classifier performance.
# 

# In[2]:


# load data from a file and append it to the rawData
def loadData(path, Text=None):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            if line[0] == "DOC_ID":  # skip the header
                continue
            (Id, rating, verified_Purchase, product_id, Text, Label) = parseReview(line) #parsereview will be read the line and return the field data
            rawData.append((Id, rating, verified_Purchase, product_id, Text, Label)) #add the data to the rawData list for every line in the file


# In[3]:


def splitData(percentage):
    # A method to split the data between trainData and testData 
    dataSamples = len(rawData)
    halfOfData = int(len(rawData)/2)
    trainingSamples = int((percentage*dataSamples)/2)
    #Seleced rating, verified purchase and product id as additional feature for the classifier. Below is the changes for adding the data in training and test data
    for (__, rating, verified_Purchase, product_id, Text, Label) in rawData[:trainingSamples] + rawData[halfOfData:halfOfData+trainingSamples]:
        trainData.append((toFeatureVector(rating, verified_Purchase, product_id,preProcess(Text)),Label))
    for (_,rating, verified_Purchase, product_id, Text, Label) in rawData[trainingSamples:halfOfData] + rawData[halfOfData+trainingSamples:]:
        testData.append((toFeatureVector(rating, verified_Purchase, product_id,preProcess(Text)),Label))


# In[4]:


# Convert line from input file into an id/rating/verified purchase/product id/text/label tuple
# Read the id, rating, verified_Purchase, product_id, Text, Label from the array indexes of the line
# Classify the true label based on the value of label in index 1 of the line
def parseReview(reviewLine):
    classname=""
    if reviewLine[1]=="__label1__":
        classname = "fake"
    else: 
        classname = "real"
    return (reviewLine[0], reviewLine[2], reviewLine[3],reviewLine[5], reviewLine[8], classname)


# # Question 4

# In order to improve the accuracy of the classifier below methods has been used:
# ### Text processing 
# - Filter the html tags with the help of beautiful soup html parser
# - Filter stopwords from the nltk.corpus library
# - Filter punctuations by creating a translator table with string.punctuation followed by   string.translate to remove the punctuations
# - Porter stem algorithm has been implemented using nltk library in order to reduce the words to their stem, for example, running, runs, runned will be trated as run being the stem of the words. Porter stem is the most simplest and widely used algorithm as per the Manning and Schutz and removes the affixes at the end of a word
# - Lemmatization is similar to stemming with a difference that the stem word is actual word in lexical decreases the feature counts from 546673 to 519620 as compared to just stemming the words. However, there is no change in scores much
# - After the filtering from the above steps are done, remaining words are converted to lower case
# - The cleaned/filtered tokens are converted to bigrams. Bigrams are set of two words/features to enable the classifier to find the probablity of the two words together during text classification
# - For weighing, the tokens are allocated a value of (1.0/len(tokens)) everytime the token is encountered
# ### Changes in SVM classifier:
# - The C parameter value has been increased from default of 1.0 to 2.5 in order to avoid the misclassifying as much as possible
# - class_weight balanced used to adjust the weights
# 
# ### Changes in classifier performance:
# Adding text preprocessing from just a simple word_tokenizer to nomalisation, weighing and removal punctuation improved the f-score and accuracy to about 2 percent from 56.7% to 58% for the testdata and 59% to 61% for the training data
# In contrast, when the TFID was added as to assign weights to the token individually with the below piece of code, the performance of the classifier was affecting. Probably, assigning more weights to the features improve the performance instead of reducing the weights
#     #TFID used for adding weights to the features for every token
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(tokens)
#     idf = vectorizer.idf_
#     localDict = dict(zip(vectorizer.get_feature_names(), idf)) 
#     featureDict.update(localDict)

# In[5]:


# TEXT PREPROCESSING AND FEATURE VECTORIZATION
# Description:
# Create a translation table creating a dictionary mapping of punctuation to none
translator = str.maketrans('', '', string.punctuation)

def preProcess(text):
    # Should return a list of tokens
    lemmatizer = WordNetLemmatizer()
    stemmer= PorterStemmer()
    filtered_tokens=[]
    lemmatized_tokens = []
    text = BeautifulSoup(text, "html.parser")    #Filter html tags
    text = text.get_text(separator=" ")
    stop_words = set(stopwords.words('english')) #Filter stop words
    text = text.translate(translator)            #Filter punctuation from the sentence
    for w in text.split(" "):                    #Text is split into words
        stemmer.stem(w)                          #Porter stemmer is used to convert the words into its stem
        if w not in stop_words:                 # Words which are not in the stopword corpus 
            filtered_tokens.append(lemmatizer.lemmatize(w.lower())) #Lemmatize all the words and convert to lower case for normalization
    filtered_tokens = [' '.join(b) for b in nltk.bigrams(filtered_tokens)] + filtered_tokens #Filtered tokens are finally converted to bigrams
    return filtered_tokens


# In[6]:


featureDict = {} # A global dictionary of features

def toFeatureVector(rating, verified_Purchase, product_id, tokens):
    # Should return a dictionary containing features as keys, and weights as values
    localDict = {}  
    
    #Rating
    if rating not in featureDict:
        featureDict[rating] = 1
    else:
        featureDict[rating] += 1
            
    if rating not in localDict:
        localDict[rating] = 1
    else:
        localDict[rating] += 1

    #Verified_Purchase
    if verified_Purchase not in featureDict:
        featureDict[verified_Purchase] = 1
    else:
        featureDict[verified_Purchase] += 1
            
    if product_id not in localDict:
        localDict[verified_Purchase] = 1
    else:
        localDict[verified_Purchase] += 1

    #Product_id
    if product_id not in featureDict:
        featureDict[product_id] = 1
    else:
        featureDict[product_id] += 1
            
    if product_id not in localDict:
        localDict[product_id] = 1
    else:
        localDict[product_id] += 1
    
    for token in tokens:
        if token not in featureDict: 
            featureDict[token] = 1 
        else:
            featureDict[token] += (1.0/len(tokens))
   
        if token not in localDict:
            localDict[token] = 1  
        else:
            localDict[token] += (1.0/len(tokens)) 
    return localDict


# In[7]:


# TRAINING AND VALIDATING OUR CLASSIFIER
def trainClassifier(trainData):
    print("Training Classifier...")
    pipeline =  Pipeline([('svc', LinearSVC(C=2.5,class_weight = 'balanced'))])
    return SklearnClassifier(pipeline).train(trainData)


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
        print(trainingSet[0])
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


# In[ ]:


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





# In[ ]:




