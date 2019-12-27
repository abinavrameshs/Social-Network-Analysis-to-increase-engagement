#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 07:36:52 2019

@author: abinavrameshsundararaman
"""


import os
os.chdir("/Users/abinavrameshsundararaman/Documents/McGill/Courses/Winter 2019/Social Media Analytics/Assignment 2")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import all libraries

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from nltk.corpus import stopwords
import nltk
from nltk.corpus import reuters
from nltk.corpus import wordnet
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from operator import itemgetter
from sklearn.metrics import classification_report
import csv
import os
import collections
import os, csv, lda, nltk
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import ast




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Scrape from Instagram 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#!/usr/bin/python3.6
import instaloader
import  time
import  pandas as pd
from datetime import datetime
from itertools import dropwhile, takewhile

L = instaloader.Instaloader()
df=pd.DataFrame()
posts = instaloader.Profile.from_username(L.context, 'natgeo').get_posts()
i=0
for post in posts:
    df = df.append({'Caption': post.caption, 'Likes': post.likes, 'Comments': post.comments, 'URL': post.url}, ignore_index=True)
    df.to_excel("Insta_withoutcomments.xlsx",index=False)
    i = i+1
    if i>1500:
        break
print("Written to Insta_withoutcomments.xlsx")



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Use Vision API to get the labels-- TASK A

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from google.cloud import vision
from google.cloud.vision import types

insta=pd.read_excel("/Users/abinavrameshsundararaman/Documents/McGill/Courses/Winter 2019/Social Media Analytics/Assignment 2/Insta_withoutcomments.xlsx")

urls=list(insta.URL)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/abinavrameshsundararaman/Downloads/Project1-06bb1889bf78.json"

client = vision.ImageAnnotatorClient()


import io

#path = '/Users/abinavrameshsundararaman/Downloads/Image.jpg'

label_list=[]
for i in urls : 
    image = vision.types.Image()
    image.source.image_uri = i
    response = client.label_detection(image=image)
    labels = response.label_annotations
    lst=[]
    for label in labels : 
        lst.append(label.description)
    label_list.append(lst)



insta['labels'] = label_list

insta.to_excel("instagram_data_with_labels.xlsx")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TASK B-Create a metric (score) for engagement by using a weighted sum of #_likes and #_comments.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

insta = pd.read_excel("instagram_data_with_labels.xlsx")
insta.columns

insta['Comments_normalized']=insta['Comments']/insta['Comments'].max()
insta['Likes_normalized']=insta['Likes']/insta['Likes'].max()


## Create engagement score column
insta['engagement_score'] = (.4*insta['Likes_normalized'])+ (.6*insta['Comments_normalized'])


## COnvert engagement score into a discrete variable
insta['engagement_score_discrete'] = (insta['engagement_score']>(insta['engagement_score'].median()))*1

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TASK B-model using image labels

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_B = TfidfVectorizer(min_df=1, 
 ngram_range=(1, 3), 
 stop_words='english', 
 strip_accents='unicode', lowercase=True, 
 norm='l2')

def content_without_stopwords_lower(text):
    stopwords = nltk.corpus.stopwords.words('english')
    wnl = nltk.WordNetLemmatizer()
    #ps = nltk.stemmer.PorterStemmer()
    content = [wnl.lemmatize(w.lower()) for w in text if w.lower() not in stopwords]
    return content


#insta["labels"]=insta["labels"].apply(lambda x :ast.literal_eval(x) )
insta["labels"]=insta["labels"].apply(lambda x :content_without_stopwords_lower(x) )
insta["labels_with_space"] = insta["labels"].apply(lambda x : " ".join(x) )


insta_vectorized = vectorizer_B.fit_transform(insta['labels_with_space'])

insta_vectorized_df = pd.DataFrame(insta_vectorized.todense())

features_B=vectorizer_B.get_feature_names()

insta_vectorized_df.columns = features_B

combined_labels_df=pd.concat([insta_vectorized_df, insta['engagement_score_discrete']], axis=1, join_axes=[insta['engagement_score_discrete'].index])


# use logistic regression to predict engagement score

X=combined_labels_df.iloc[:,:-1]
y=combined_labels_df.loc[:,['engagement_score_discrete']]


#import statsmodels.api as sm
#logit_model=sm.Logit(y,X)
#result=logit_model.fit()
#print(result.summary2())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print(accuracy_score(y_test,y_pred))
confusion_matrix(y_test,y_pred)

#
#0.7479838709677419
#Out[400]: 
#array([[189,  50],
#       [ 75, 182]])
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TASK B-model using captions

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

## Pre-Process captions

insta["Caption_token"] = insta["Caption"].apply(nltk.word_tokenize)

def include_only_alphas(list1):
    l=list()
    for i in list1 : 
        if(i.isalpha()):
            l.append(i)
    return l

insta["Caption_token"] = insta["Caption_token"].apply(include_only_alphas)

def preprocess(text):
    stopwords = nltk.corpus.stopwords.words('english')
    wnl = nltk.WordNetLemmatizer()
    content = [wnl.lemmatize(w.lower()) for w in text if (w.lower() not in stopwords)]
    return content

insta["Caption_token"]=insta["Caption_token"].apply(lambda x :content_without_stopwords_lower(x) )

insta["Caption_with_space"] = insta["Caption_token"].apply(lambda x : " ".join(x) )

vectorizer_B1 = TfidfVectorizer(min_df=1, 
 ngram_range=(1, 3), 
 stop_words='english', 
 strip_accents='unicode', lowercase=True, 
 norm='l2')


insta_caption_vectorized = vectorizer_B1.fit_transform(insta['Caption_with_space'])

insta_caption_vectorized_df = pd.DataFrame(insta_caption_vectorized.todense())

features_B1=vectorizer_B1.get_feature_names()

insta_caption_vectorized_df.columns = features_B1

combined_captions_df=pd.concat([insta_caption_vectorized_df, insta['engagement_score_discrete']], axis=1, join_axes=[insta['engagement_score_discrete'].index])


# use logistic regression to predict engagement score

X1=combined_labels_df.iloc[:,:-1]
y1=combined_labels_df.loc[:,['engagement_score_discrete']]



X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.33)

logreg1 = LogisticRegression()
logreg1.fit(X_train1, y_train1)

y_pred1 = logreg1.predict(X_test1)
print(accuracy_score(y_test1,y_pred1))
confusion_matrix(y_test1,y_pred1)

#0.7419354838709677
#Out[399]: 
#array([[195,  55],
#       [ 73, 173]])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TASK B-model using both captions and Image labels

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# COmbine label and caption predictors

combined_labels_captions_df=pd.concat([insta_vectorized_df,insta_caption_vectorized_df, insta['engagement_score_discrete']], axis=1, join_axes=[insta['engagement_score_discrete'].index])


# use logistic regression to predict engagement score

X2=combined_labels_captions_df.iloc[:,:-1]
y2=combined_labels_captions_df.loc[:,['engagement_score_discrete']]



X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.33)

logreg2 = LogisticRegression()
logreg2.fit(X_train2, y_train2)

y_pred2 = logreg2.predict(X_test2)
print(accuracy_score(y_test2,y_pred2))
confusion_matrix(y_test2,y_pred2)

#0.7661290322580645
#Out[398]: 
#array([[187,  66],
#       [ 50, 193]])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TASK C- Topic Modelling : LDA

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

insta.columns
insta_LDA_df=insta.loc[:,["engagement_score","labels_with_space"]]

#checking for nulls if present any
print("Number of rows with any of the empty columns:")
print(insta_LDA_df.isnull().sum().sum())
insta_LDA_df=insta_LDA_df.dropna() 

engagement_score = 'engagement_score'
labels = 'labels_with_space'
ntopics= 4

word_tokenizer=RegexpTokenizer(r'\w+')
wordnet_lemmatizer = WordNetLemmatizer()
stopwords_nltk=set(stopwords.words('english'))

def tokenize_text(version_desc):
    lowercase=version_desc.lower()
    text = wordnet_lemmatizer.lemmatize(lowercase)
    tokens = word_tokenizer.tokenize(text)
    return tokens

vec_words = CountVectorizer(tokenizer=tokenize_text,stop_words=stopwords_nltk,decode_error='ignore')
total_features_words = vec_words.fit_transform(insta_LDA_df[labels])

print(total_features_words.shape)

model = lda.LDA(n_topics=int(ntopics), n_iter=500, random_state=1)
model.fit(total_features_words)

topic_word = model.topic_word_ 
doc_topic=model.doc_topic_
doc_topic=pd.DataFrame(doc_topic)
insta_LDA_df=insta_LDA_df.join(doc_topic)
engagement=pd.DataFrame()

for i in range(int(ntopics)):
    topic="topic_"+str(i)
    engagement[topic]=insta_LDA_df.groupby([engagement_score])[i].mean()
    
engagement=engagement.reset_index()
topics=pd.DataFrame(topic_word)
topics.columns=vec_words.get_feature_names()
topics_df=topics.transpose()
topics_df.to_excel("topic_word_dist.xlsx")
engagement.to_excel("engagement_topic_dist.xlsx",index=False)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TASK C- Compare the distribution of topics between high and low engagement scores

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## Naming of each topic present

TOPIC 0 : 
    
plant
wildlife
tree
animal
terrestrial
nature


TOPIC 1 : 
  
sky
water
phenomenon
mountain
landscape
natural
    

    
TOPIC 2 : 

marine
seal
architecture
water
mammal

TOPIC 3 : 

bird
wildlife
vertebrate
bear
animal
turtle
terrestrial
penguin


TOPIC 4 : 

photography
black
monochrome
white
human
people
event
vehicle
adaptation


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Compare the distribution of topics between high and low engagement scores

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

engagement_analysis = engagement


# Divide engagement scores into quantiles
engagement_analysis.engagement_score.describe()['mean']
engagement_analysis.engagement_score.describe()['25%']
engagement_analysis.engagement_score.describe()['50%']
engagement_analysis.engagement_score.describe()['75%']

def split_into_quartiles(y):
    if y<engagement_analysis.engagement_score.describe()['25%'] : 
        return(1)
    elif engagement_analysis.engagement_score.describe()['25%']<=y<engagement_analysis.engagement_score.describe()['50%'] : 
        return(2)
    elif engagement_analysis.engagement_score.describe()['50%']<=y<engagement_analysis.engagement_score.describe()['75%'] : 
        return(2)
    else : 
        return(3)
        
engagement_analysis['quartile'] =engagement_analysis["engagement_score"].apply(lambda x : split_into_quartiles(x))

### 

temp=engagement_analysis.groupby(['quartile']).mean().reset_index()



###################



# first quartile
first_quartile=engagement_analysis[engagement_analysis.quartile==1]

bins = np.linspace(0, 1, 100)

plt.hist(first_quartile.topic_0, bins, alpha=0.5, label='topic_0')
plt.hist(first_quartile.topic_1, bins, alpha=0.5, label='topic_1')
plt.hist(first_quartile.topic_2, bins, alpha=0.5, label='topic_2')
plt.hist(first_quartile.topic_3, bins, alpha=0.5, label='topic_3')
plt.hist(first_quartile.topic_4, bins, alpha=0.5, label='topic_4')
plt.legend(loc='upper right')
plt.show()

# second quartile
second_quartile=engagement_analysis[engagement_analysis.quartile==2]

bins = np.linspace(0, 1, 100)

plt.hist(second_quartile.topic_0, bins, alpha=0.5, label='topic_0')
plt.hist(second_quartile.topic_1, bins, alpha=0.5, label='topic_1')
plt.hist(second_quartile.topic_2, bins, alpha=0.5, label='topic_2')
plt.hist(second_quartile.topic_3, bins, alpha=0.5, label='topic_3')
plt.hist(second_quartile.topic_4, bins, alpha=0.5, label='topic_4')
plt.legend(loc='upper right')
plt.show()

# third quartile
third_quartile=engagement_analysis[engagement_analysis.quartile==3]

bins = np.linspace(0, 1, 100)

plt.hist(third_quartile.topic_0, bins, alpha=0.5, label='topic_0')
plt.hist(third_quartile.topic_1, bins, alpha=0.5, label='topic_1')
plt.hist(third_quartile.topic_2, bins, alpha=0.5, label='topic_2')
plt.hist(third_quartile.topic_3, bins, alpha=0.5, label='topic_3')
plt.hist(third_quartile.topic_4, bins, alpha=0.5, label='topic_4')
plt.legend(loc='upper right')
plt.show()

# fourth quartile
fourth_quartile=engagement_analysis[engagement_analysis.quartile==4]

bins = np.linspace(0, 1, 100)

plt.hist(fourth_quartile.topic_0, bins, alpha=0.5, label='topic_0')
plt.hist(fourth_quartile.topic_1, bins, alpha=0.5, label='topic_1')
plt.hist(fourth_quartile.topic_2, bins, alpha=0.5, label='topic_2')
plt.hist(fourth_quartile.topic_3, bins, alpha=0.5, label='topic_3')
plt.hist(fourth_quartile.topic_4, bins, alpha=0.5, label='topic_4')
plt.legend(loc='upper right')
plt.show()




sns.distplot(first_quartile.topic_0, hist=False, kde=True,bins=int(180/5), color = 'darkblue', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})
sns.distplot(first_quartile.topic_1, hist=False, kde=True,bins=int(180/5), color = 'red', hist_kws={'edgecolor':'red'},kde_kws={'linewidth': 2})
sns.distplot(first_quartile.topic_2, hist=False, kde=True,bins=int(180/5), color = 'green', hist_kws={'edgecolor':'red'},kde_kws={'linewidth': 2})
sns.distplot(first_quartile.topic_3, hist=False, kde=True,bins=int(180/5), color = 'blue', hist_kws={'edgecolor':'red'},kde_kws={'linewidth': 2})
sns.distplot(first_quartile.topic_4, hist=False, kde=True,bins=int(180/5), color = 'orange', hist_kws={'edgecolor':'red'},kde_kws={'linewidth': 2})
