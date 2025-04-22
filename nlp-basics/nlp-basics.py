# %% [markdown]
# # nlp-basics

# %%
# imports

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import spacy
pd.set_option('display.max_colwidth', None)

# %% [markdown]
# ### Exercise: use the description feature of the dataset and complete the following tasks:
#
# - Tokenize the description feature
# - Remove stop words
# - Stem the tokens
# - Lemmatize the tokens
# - Create two new features: "Cleaned_Stem_Description" and "Cleaned_Lemma_Description"
# - Build a word cloud based on Cleaned_Lemma_Description

# %%
# import the data

data = pd.read_csv('winemag-data.csv')
data.head()

# %%
# check for empty lists

data[data['description'] == '']

# %%
# check duplicates

data[['description','taster_name']].duplicated().sum()

# %%
# drop duplicates

data = data.drop_duplicates(subset=['description','taster_name'])

# %%
# check duplicates

data[['description','taster_name']].duplicated().sum()

# %%
# only get the data I need, the description feature

df = data[['description']].copy()
df.head()

# %%
# check for nulls

df.isna().sum()

# %%
df

# %%
# check for urls

df.loc[df['description'].str.contains('www' or 'http')]

# %%
# change everything to lowercase

df['lower'] = df['description'].str.lower()
df

# %% [markdown]
# # 1. Tokenize the description feature

# %%
# Tokenize the description feature

df['basic_tokens'] = df['lower'].str.split()
df.head()

# %% [markdown]
# # 2. Remove stop words

# %%
# download stop words from nltk

nltk.download('stopwords')

# get the list of english stop words from the corpus

stopwords = nltk.corpus.stopwords.words('english')

# %%
# create a function to remove stop words using list comprehension

def remove_stopwords(text):
  return [word for word in text if word not in stopwords]

df['no_stop'] = df['basic_tokens'].apply(remove_stopwords)

# %%
# check the results

df.head()

# %% [markdown]
# ### Remove punctuation too

# %%
# get the punctuation

from string import punctuation

# %%
# create a list of punctuations

punc = list(punctuation)

# %%
# create a function to remove the punctuation

def remove_punc(text):
  return [word for word in text if word not in punc]

# apply the function to remove punctuation and create a new column

df['no_punc'] = df['no_stop'].apply(remove_punc)
df.head()

# %% [markdown]
# # 3. Stem the tokens

# %%
# import PorterStemmer

from nltk import PorterStemmer

# define the stemmer

stemmer = PorterStemmer()

# %%
# create a function to use the stemmer to get the stem of a word

def stemmer_func(text):
  return [stemmer.stem(word) for word in text]

# %%
# apply the function to create a new column

df['Cleaned_Stem_Description'] = df['no_punc'].apply(stemmer_func)
df

# %% [markdown]
# # 4. Lemmatize the tokens

# %%
# import spacy

import spacy

# %%
# create the model

nlp_model = spacy.load('en_core_web_sm', disable=('parser','ner'))

# %%
# define a function for the spacy process that includes lemmatization

def spacy_process(text):
  """
  nlp_model must be already globally defined
  """

  doc = nlp_model(text)
  processed_doc = [token.lemma_.lower() for token in doc if
                   not token.is_stop and
                   not token.is_punct and
                   not token.is_space]

  return processed_doc

# %%
# apply the function

df['Cleaned_Lemma_Description'] = df['description'].apply(spacy_process)
df

# %% [markdown]
# # 5. Build a word cloud based on Cleaned_Lemma_Description

# %%
# import wordcloud

from wordcloud import WordCloud

# %%
# make all the words into 1 string

words_string = df['Cleaned_Lemma_Description'].explode().astype(str).to_list()
words_string = ' '.join(words_string)

# %%
# generate a wordcloud

word_cloud = WordCloud(min_word_length=2).generate(words_string)

# %%
# plot the wordcloud

plt.imshow(word_cloud)
plt.title('Words in the Wine Review Dataset')
plt.axis('off')

# %% [markdown]
# # NLP Pipeline

# %%
from tqdm import tqdm

# %%
# define a function for the nlp pipeline

def spacy_process_pipeline(text):
  processed = []
  for doc in tqdm(nlp_model.pipe(text, batch_size=1000)):
    lemmas = [token.lemma_.lower() for token in doc if
              not token.is_stop and
              not token.is_punct and
              not token.is_space]

    processed.append(lemmas)

  return processed

# %%
# use the pipeline to create new column

df['spacy_pipe'] = spacy_process_pipeline(df['description'])

# %%
# check the data

df

# %%
