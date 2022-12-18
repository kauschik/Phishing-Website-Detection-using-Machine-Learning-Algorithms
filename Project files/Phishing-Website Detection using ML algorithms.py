#!/usr/bin/env python
# coding: utf-8

# # **Phishing-Website Detection Using Machine Learning Algorithms**
# 

# **Project by :** *Narra SuryaKoushik Reddy*
# 
# **E-mail :** *narrasuryakoushikreddy@gmail.com*
# 
# **Project for :** *Mini project for Data Science in JNTU Hyderabad*

# ## 1. Installing & Importing Useful libraries

# There will be some libraries which are not preinstall, we need to install them manually inorder to run this program

# In[29]:


get_ipython().system('pip install selenium')


# In[30]:


get_ipython().system('pip install wordcloud')


# In[31]:


import pandas as pd
import numpy as np 

import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import time 

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB 

from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.pipeline import make_pipeline 

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from bs4 import BeautifulSoup 
from selenium import webdriver  
import networkx as nx 
import pickle

import warnings
warnings.filterwarnings('ignore')


# ## 2. Loading and Analysing the Dataset

# In[32]:


# Lading the data set
data_OG = pd.read_csv('phishing_site_urls.csv')


# In[86]:


print("First 5 columns of the data")
print(data_OG.head())
print('-'*70)
print("Last 5 columns of the data")
print(data_OG.tail())


# In[34]:


# Information of the Dataset
print(data_OG.info())


# In[35]:


# Checking for missiong values
print(data_OG.isnull().sum())


# 

# Let's check to determine if the classes are balanced or imbalanced if there are classification issues.

# In[36]:


# Class count dataframe
label_counts = pd.DataFrame(data_OG.Label.value_counts())


# In[37]:


#visualizing target_col
fig = px.bar(label_counts, x=label_counts.index, y=label_counts.Label)
fig.show()


# ## 3. Preprocessing 

# As soon as we obtain the information, we must vectorize our URLs. Since some terms in urls, such as "virus," ".exe," and ".dat," are more essential than others, I utilised CountVectorizer and gathered words using tokenizer. Let's create a vector form using the URLs.

# ### 3.1 RegexpTokenizer

# A tokenizer that divides a text into tokens and separators by using a regular expression that matches both.

# In[38]:


tokenizer = RegexpTokenizer(r'[A-Za-z]+')#getting alpha only


# In[39]:


data_OG.URL[0]


# In[40]:


# pull letter which matches to expression
tokenizer.tokenize(data_OG.URL[0]) # using first row


# In[41]:


print('Getting words tokenized ...')
t0= time.perf_counter()
data_OG['text_tokenized'] = data_OG.URL.map(lambda t: tokenizer.tokenize(t)) # doing with all rows
t1 = time.perf_counter() - t0
print('Time taken',t1 ,'sec')


# In[42]:


data_OG.sample(5)


# ### 3.2 SnowballStemmer

# A simple string processing language called Snowball provides root words.

# In[43]:


stemmer = SnowballStemmer("english")


# In[44]:


print('Getting words stemmed ...')
t0= time.perf_counter()
data_OG['text_stemmed'] = data_OG['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')


# In[45]:


data_OG.sample(5)


# In[46]:


print('Getting joiningwords ...')
t0= time.perf_counter()
data_OG['text_sent'] = data_OG['text_stemmed'].map(lambda l: ' '.join(l))
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')


# In[47]:


data_OG.sample(5)


# # 4. Visualization

# Use a word cloud to visualise certain crucial keys.

# In[48]:


#sliceing classes
bad_sites = data_OG[data_OG.Label == 'bad']
good_sites = data_OG[data_OG.Label == 'good']


# In[85]:


print("First 5 columns of bad_sites")
print(bad_sites.head())
print('-'*70)
print("First 5 columns of good sites")
print(good_sites.tail())


# - make a function that displays the key elements from the url

# In[50]:


def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'com','http'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = mask)
    wordcloud.generate(text)
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
d = '../input/masks/masks-wordclouds/'


# In[51]:


data = good_sites.text_sent
data.reset_index(drop=True, inplace=True)


# In[52]:


data = bad_sites.text_sent
data.reset_index(drop=True, inplace=True)


# # 5. Creating models 

# ### 5.1 CountVectorizer

# - A corpus of text is converted into a vector of term/token counts using the CountVectorizer.

# In[53]:


#Create object
cv = CountVectorizer()


# In[54]:


help(CountVectorizer())


# In[55]:


#transforming all text which we tokenize and stemed
feature = cv.fit_transform(data_OG.text_sent)


# In[56]:


# convert sparse matrix into array to print transformed features
feature[:5].toarray() 


# ### 5.2 Splitting the data

# In[58]:


trainX, testX, trainY, testY = train_test_split(feature, data_OG.Label)


# ### LogisticRegression

# An approach for machine learning called logistic regression is used to predict the likelihood of a categorical dependent variable. The dependent variable in logistic regression is a binary variable with data coded as 1 (yes, success, etc.) or 0 (no) (no, failure, etc.). In other words, P(Y=1) is predicted by the logistic regression model as a function of X.

# In[59]:


# create lr object
lr = LogisticRegression()


# In[60]:


lr.fit(trainX,trainY)


# In[61]:


lr.score(testX,testY)


# - 
# Logistic Regression is giving 96% accuracy, Now we will store scores in dict to see which model perform best

# In[63]:


Scores_ml = {}
Scores_ml['Logistic Regression'] = np.round(lr.score(testX,testY),2)


# In[64]:


print('Training Accuracy :',lr.score(trainX,trainY))
print('Testing Accuracy :',lr.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(lr.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# ### MultinomialNB

# - NLP Problems Using Multinomial Naive Bayes. A series of probabilistic algorithms known as the Naive Bayes Classifier Algorithm is based on using the Bayes Theorem with the "naive" assumption that each pair of features is conditionally independent.

# In[65]:


# create mnb object
mnb = MultinomialNB()


# In[66]:


mnb.fit(trainX,trainY)


# In[67]:


mnb.score(testX,testY)


# - MultinomialNB is giving us 95% accuracy  

# In[68]:


Scores_ml['MultinomialNB'] = np.round(mnb.score(testX,testY),2)


# In[69]:


print('Training Accuracy :',mnb.score(trainX,trainY))
print('Testing Accuracy :',mnb.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(mnb.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(mnb.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[70]:


acc = pd.DataFrame.from_dict(Scores_ml,orient = 'index',columns=['Accuracy'])
sns.set_style('darkgrid')
sns.barplot(acc.index,acc.Accuracy)


# -  So, Logistic Regression is the best fit model, Now we make sklearn pipeline using Logistic Regression

# In[71]:


pipeline_ls = make_pipeline(CountVectorizer(tokenizer = 
                                            RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())
##(r'\b(?:http|ftp)s?://\S*\w|\w+|[^\w\s]+') ([a-zA-Z]+)([0-9]+)  -- these tolenizers giving me low accuray 


# In[75]:


trainX, testX, trainY, testY = train_test_split(data_OG.URL, data_OG.Label)


# In[76]:


pipeline_ls.fit(trainX,trainY)


# In[78]:


pipeline_ls.score(testX,testY)


# In[79]:


print('Training Accuracy :',pipeline_ls.score(trainX,trainY))
print('Testing Accuracy :',pipeline_ls.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(pipeline_ls.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(pipeline_ls.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[80]:


pickle.dump(pipeline_ls,open('phishing.pkl','wb'))


# In[81]:


loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.score(testX,testY)
print(result)


# - I'm done now. See how easy it is yet how powerful it is. A 98% accuracy rate is obtained. That is a fairly high threshold at which a system may identify a bad URL. Want to test a few links to see if the model can accurately predict the future? Sure. Let's proceed.

# In[ ]:




