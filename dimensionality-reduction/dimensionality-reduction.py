# %% [markdown]
# # Dimensionality Reduction Exercise

# %% [markdown]
# Information about the dataset:
#
# - low res handwritten numerals
# - each row is 1 handwritten numeral
# - each column is a pixel
# - images are 28 x 28
# - take the pixels and predict which number it is
#
# Additional instructions:
#
# - try to go further
# - use pca as a transformer in a pipeline
# - to reduce dimensionality
# - see how much you can reduce the size and still get a good score

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn import set_config
set_config(transform_output='pandas')

# %%
# import data

df = pd.read_csv('mnist.csv')

# %%
# check the data

df.head()

# %%
# tips about imshow

image = df.iloc[0,1:].values.reshape(28, 28)
plt.imshow(image)   # shows the image

# %% [markdown]
# ### 1) Perform PCA on the data in mnist.csv for dimensionality reduction. Show the components that retain 90% of the variance.

# %%
# check for nulls

df.isna().sum().sum()

# %%
# check dtypes

df.dtypes.nunique()

# %%
# drop the label

df_drop = df.drop(columns='label')
df_drop.head()

# %%
# scale the data

df_scaled = StandardScaler().fit_transform(df_drop)
df_scaled

# %%
# define the model to retain 90% of the variance

pca = PCA(0.9, random_state=42)
pca_data = pca.fit_transform(df_scaled)

# %%
# check how many components are left

pca_data.shape

# %%
# plot the explained variance ratios

plt.plot(pca.explained_variance_ratio_)

# %%
# plot the cumulative variance as the number of n_components increases

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Cumulative Explained Variance')
plt.grid(True)
plt.axhline(y=0.9, color='r')
plt.show()


# %% [markdown]
# ### 2) Use PCA in a pipeline and reduce dimensionality as much as possible while preserving score

# %%
# make a pipe line using KNN and PCA

knn = KNeighborsClassifier()

knn_pca_pipe = make_pipeline(StandardScaler(), pca, knn)

# %%
# fit the model

X = df.drop(columns=['label'])
y = df['label'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_pca_pipe.fit(X_train, y_train)

# %%
# check the scores

scores = pd.DataFrame()
scores.loc['n_components = 0.9','test_accuracy'] = knn_pca_pipe.score(X_test, y_test)
scores.loc['n_components = 0.9','train_accuracy'] = knn_pca_pipe.score(X_train, y_train)
scores

# %%
df.shape[1]

# %%
# create a loop to try different n_components
import time

n_range = range(2, df.shape[1], 10)

score_loop = pd.DataFrame()

for n in n_range:
  print(f'Trying n={n}')
  pca_n = PCA(n_components=n, random_state=42)
  pca_data = pca_n.fit_transform(df_scaled)

  start_fit = time.time()
  knn_pca_n_pipe = make_pipeline(StandardScaler(), pca_n, knn).fit(X_train, y_train)
  end_fit = time.time()

  score_loop.loc[f'n_components = {n}','n'] = n
  start_predict = time.time()
  score_loop.loc[f'n_components = {n}','test_accuracy'] = knn_pca_n_pipe.score(X_test, y_test)
  end_predict = time.time()
  score_loop.loc[f'n_components = {n}','train_accuracy'] = knn_pca_n_pipe.score(X_train, y_train)
  score_loop.loc[f'n_components = {n}','fit_time'] = end_fit - start_fit
  score_loop.loc[f'n_components = {n}','predict_time'] = end_predict - start_predict

# %%
score_loop

# %%
score_loop.sort_values(by='train_accuracy')

# %%
# plot train accuracy

plt.figure(figsize=(20,5))
plt.plot(score_loop['train_accuracy'])
plt.xticks(rotation=45)
plt.title('Train Accuracy with 2-782 n_components');

# %%
# plot test accuracy

plt.figure(figsize=(20,5))
plt.plot(score_loop['test_accuracy'])
plt.xticks(rotation=45)
plt.title('Test Accuracy with 2-782 n_components');

# %%
# plot the times

plt.figure(figsize=(20,5))
plt.plot(score_loop[['fit_time','predict_time']])
plt.xticks(rotation=45)
plt.legend(['fit_time', 'predict_time'])
plt.title('Fitting and Predicting Times with n_components 2 to 782');

# %%
# create a new loop with smaller step and narrowed in on the elbow

n_range = range(2, 50, 1)

score_loop_2 = pd.DataFrame()

for n in n_range:
  print(f'Trying n={n}')
  pca_n = PCA(n_components=n, random_state=42)
  pca_data = pca_n.fit_transform(df_scaled)

  start_fit = time.time()
  knn_pca_n_pipe = make_pipeline(StandardScaler(), pca_n, knn).fit(X_train, y_train)
  end_fit = time.time()

  score_loop_2.loc[f'n_components = {n}','n'] = n
  start_predict = time.time()
  score_loop_2.loc[f'n_components = {n}','test_accuracy'] = knn_pca_n_pipe.score(X_test, y_test)
  end_predict = time.time()
  score_loop_2.loc[f'n_components = {n}','train_accuracy'] = knn_pca_n_pipe.score(X_train, y_train)
  score_loop_2.loc[f'n_components = {n}','fit_time'] = end_fit - start_fit
  score_loop_2.loc[f'n_components = {n}','predict_time'] = end_predict - start_predict

# %%
# check the scores

score_loop_2

# %%
# plot train accuracy

plt.figure(figsize=(10,5))
plt.plot(score_loop_2['train_accuracy'])
plt.xticks(rotation=45)
plt.title('Train Accuracy with 2-49 n_components')
plt.axvline(x='n_components = 15', color='r');

# %%
# plot test accuracy

plt.figure(figsize=(10,5))
plt.plot(score_loop_2['test_accuracy'])
plt.xticks(rotation=45)
plt.title('Test Accuracy with 2-49 n_components')
plt.axvline(x='n_components = 15', color='r', label="n_components=15");

# %%
