#!/usr/bin/env python
# coding: utf-8

# # BODYSHAMING TWEET ANALYSIS 

# ### IMPORTING THE REQUIRED PACKAGES AND MODULES

# In[1]:


import pandas as pd              #for data analysis and basic operations
import numpy as np               #for data analysis and basic operations
import re                        #for regex
import seaborn as sns            #for data visualisation
import matplotlib.pyplot as plt  #for data visualisation
from matplotlib import style     #style for the plot
style.use('ggplot')
from textblob import TextBlob                    #process the textual data
from nltk.tokenize import word_tokenize          #for tokenization
from nltk.stem import PorterStemmer              #for stemming
from nltk.corpus import stopwords                #to remove stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import CountVectorizer                          #to vectorize the text document
from sklearn.model_selection import train_test_split                                 #to split the data into training and testing data
from sklearn.linear_model import LogisticRegression                                  #to perform logistic regression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay # for evaluating accuracy and displaying matrix for evaluating the model


# ### READING  AND DISPLAYING THE DATASET

# In[2]:


#read the csv file
df = pd.read_csv('tweets.csv',encoding='latin-1',header=None)


# In[3]:


#display the first 10 entries
df.head(10)


# In[4]:


df.info()          #to describe the dataset


# In[5]:


df.columns        #to obtain the column names


# In[6]:


#renaming the columns
df = df.rename(columns = {0: 'polarity', 1: 'IDs', 2: 'date', 3: 'flag', 4: 'username', 5: 'text'})


# In[7]:


#display the new dataframe
df.head()


# ### CREATING A NEW DATAFRAME FOR TWEETS

# In[8]:


text_df = df.drop(['polarity','IDs', 'date', 'flag', 'username'], axis=1) #drop all columns except the "text" column
text_df.head(10)                                   #new dataframe


# In[9]:


print(text_df['text'].iloc[0],"\n")   #analyse data in the "text" dataframe
print(text_df['text'].iloc[1],"\n")
print(text_df['text'].iloc[2],"\n")
print(text_df['text'].iloc[3],"\n")
print(text_df['text'].iloc[4],"\n")


# In[10]:


text_df.info() #to describe the new dataframe


# ### CONVERSION OF RAW DATA TO USEFUL DATA

# In[11]:


def data_processing(text):            #to convert the raw data into usable format
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)  #remove URLs
    text = re.sub(r'\@w+|\#','',text) #remove hashtags 
    text = re.sub(r'[^\w\s]','',text) #remove punctuation marks
    text_tokens = word_tokenize(text) #remove stopwords
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


# In[12]:


text_df.text = text_df['text'].apply(data_processing) 


# In[13]:


text_df = text_df.drop_duplicates('text')  #remove duplicate data


# ### STEMMING 

# In[14]:


stemmer = PorterStemmer()  #stemming for reducing tokenized words to their root form
def stemming(data):
    tweet = [stemmer.stem(word) for word in data]
    return data


# In[15]:


text_df['text'] = text_df['text'].apply(lambda x: stemming(x)) #apply stemming to the processed data


# In[16]:


text_df.head(10)


# In[17]:


print(text_df['text'].iloc[0],"\n") 
print(text_df['text'].iloc[1],"\n")
print(text_df['text'].iloc[2],"\n")
print(text_df['text'].iloc[3],"\n")
print(text_df['text'].iloc[4],"\n")


# In[18]:


text_df.info() #updated dataframe


# ### CALCULATING THE POLARITY

# In[19]:


def polarity(text):                          #to calculate polarity using TextBlob
    return TextBlob(text).sentiment.polarity


# In[20]:


text_df['polarity'] = text_df['text'].apply(polarity)


# In[21]:


text_df.head(10)


# ### OBTAINING THE SENTIMENT LABEL FOR EACH TWEET

# In[22]:


def sentiment(label):     #to define the sentiment of a particular tweet
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"


# In[23]:


text_df['sentiment'] = text_df['polarity'].apply(sentiment)


# In[24]:


text_df.head(20)


# ### VISUALIZATION OF DATA USING COUNTPLOT AND PIE CHART

# In[25]:


fig = plt.figure(figsize=(5,5))                #data visualization using countplot
sns.countplot(x='sentiment', data = text_df)


# In[26]:


fig = plt.figure(figsize=(7,7))                #data visualization using pie chart
colors = ("yellow", "limegreen", "purple")
wp = {'linewidth':2, 'edgecolor':"black"}
tags = text_df['sentiment'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,startangle=90, wedgeprops = wp, explode = explode, label='')
plt.title('*** VISUALIZATION OF SENTIMENTS ***')


# ### BUILDING THE MODEL

# In[27]:


vect = CountVectorizer(ngram_range=(1,2)).fit(text_df['text'])  #count vectorization for the model


# In[28]:


feature_names = vect.get_feature_names()                        #get and print the first 30 features
print("Total number of features are: {}\n".format(len(feature_names)))
print("The first 100 features are:\n {}".format(feature_names[:100]))


# In[29]:


X = text_df['text']                 #separation of data into x and y for transformation
Y = text_df['sentiment']
X = vect.transform(X)


# In[30]:


#split the data into training and testing data

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[31]:


#print the size of training and testing data

print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))

print("Size of x_test:", (x_test.shape))
print("Size of y_test:", (y_test.shape))


# In[32]:


#to get rid of warnings

import warnings
warnings.filterwarnings('ignore')


# ### TRAINING THE MODEL

# In[33]:


#train the data on logisticregression model

logreg = LogisticRegression()

logreg.fit(x_train, y_train) #fit the data
logreg_pred = logreg.predict(x_test) #predict the value for test data

logreg_acc = accuracy_score(logreg_pred, y_test) #calculate the accuracy for the model
print("Accuracy of the model is: {:.2f}%".format(logreg_acc*100)) 


# ### OBTAINING THE CLASSIFICATION REPORT AND PRINTING THE RELEVANT CONFUSION MATRIX

# In[34]:


#display the confusion matrix

style.use('classic')
cm = confusion_matrix(y_test, logreg_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=logreg.classes_)
disp.plot()


# In[35]:


#print the confusion matrix and classification report


print(confusion_matrix(y_test, logreg_pred))
print("\n")
print(classification_report(y_test, logreg_pred))


# In[36]:


y_pred = logreg.predict(x_train)
accuracy2 = accuracy_score(y_train, y_pred)
print("Accuracy of the model for training set is:", accuracy2)


# In[37]:


#print the confusion matrix and classification report


print(confusion_matrix(y_train, y_pred))
print("\n")
print(classification_report(y_train, y_pred))


# In[38]:


#display the confusion matrix

style.use('classic')
cm2 = confusion_matrix(y_train, y_pred, labels=logreg.classes_)
disp2 = ConfusionMatrixDisplay(confusion_matrix = cm2, display_labels=logreg.classes_)
disp2.plot()


# In[ ]:




