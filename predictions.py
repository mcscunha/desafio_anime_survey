#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip install -U pip numpy pandas matplotlib seaborn scikit-learn --no-cache-dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import path

anime = pd.read_csv('./db/anime_filtered.csv')
anime.sample(5)

anime.shape
anime.info()
anime.describe()
anime.describe( include='O')

anime.columns

# premiered
anime["isPremiered"] = anime["premiered"].isnull().astype(int)

anime.isPremiered.info()
# no nulls

# for studion replace the null with unknown
anime["studio"] = anime["studio"].fillna("unknown")

studion_counts = anime.studio.value_counts()
studion_counts

minor_classes = studion_counts[studion_counts < 40].index.to_list()
minor_classes

anime["studio"] = anime["studio"].apply(lambda x: "small_studio" if x in minor_classes else x)
anime.studio.value_counts()
anime.columns
anime.drop(
    columns=[
        'anime_id',
        'title_english',
        'title_japanese',
        'title_synonyms',
        'image_url',
        'background',
        'premiered',
        'broadcast',
        'producer',
        'licensor',
        'related',
        'opening_theme',
        'ending_theme',
        'aired_string',
        'aired'
    ],
    inplace=True
)
anime.shape
anime.columns

anime.genre = anime.genre.fillna("notCat")
genre_df = anime['genre'].str.get_dummies(sep=',')
genre_df.sample(10)
genre_df.info()

anime_df = pd.concat([anime, genre_df], axis=1)
anime_df
anime_df.drop(columns=["genre"], inplace=True)
anime_df

anime_df.describe(include='O')
# I used this function to find the objects still in the code that need to be encodeed and the number of classes at each

string_col = anime_df.select_dtypes(include='O').columns
string_col
for col in string_col:
    anime_df[col] = anime_df[col].astype('category').cat.codes

anime_df.select_dtypes(include='O').columns

anime_df
# we had some boolean

anime_df.select_dtypes(include='bool')
# only this column

anime_df['airing'] = anime_df['airing'].astype('category').cat.codes
anime_df['airing']
# perfecto

anime_df.shape

anime_df.drop(columns=['title'], inplace=True)

# let's make sure no null values
for col in anime_df:
    print(f" {col}         | has ({anime_df[col].isnull().sum()})")

anime_df.rating.value_counts()
anime_df.columns

anime_df['rank'] = anime_df['rank'].fillna(anime_df['rank'].max())
# if you are not rated who care ⁉️😀 except your fans
anime_df
# to scale the Scored_by column and rank
anime_df['rank'] /= 1000
anime_df['scored_by'] /= 1000

anime_df.sample(3)




from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

X = anime_df.loc[:, ~ anime_df.columns.isin(['score'])]

y = anime_df['score']

X_train, X_test,y_train, y_test = tts(X,y, test_size=0.3, random_state=42)

X_train.shape

X_test.shape

# ## Building four regression models
# - Deicsion tree
# - Random forest
# - Gradient Boossting
# - Extra Trees
# > Related notes
# * Bootsrapping , Bagging and Boosting
# https://hossam-ahmed.notion.site/The-Three-Bs-Bootstrapping-Bagging-Boosting-ebbac979d168423a90d1fe552b6d0def?pvs=4
# * Entropy , information gain and gini index
# https://hossam-ahmed.notion.site/Entropy-Information-Gain-Gini-Index-1fbaf1424fe8495f91b68d11a071a930?pvs=4
# * cosine similarity
# https://hossam-ahmed.notion.site/Cosine-similarity-54f8e1b4e93642e7a3c4fd887d826400?pvs=4
# * Ensemble contains details about [Extra trees](!https://www.notion.so/hossam-ahmed/Ensemble-methods-scikit-learn-0361f6989cc645fcac2573fddd97f6d7?pvs=4#29e4688c6a3e48978a457dc6a31a2c81)
# https://hossam-ahmed.notion.site/Articles-4d8f0c0a7af84bbb95816cc867e2163c?pvs=4

dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
et = ExtraTreesRegressor(n_estimators=100, random_state=42)

dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
et.fit(X_train, y_train)

dt_preds = np.clip(dt.predict(X_test), 0, 10)
rf_preds = np.clip(rf.predict(X_test), 0, 10)
gb_preds = np.clip(gb.predict(X_test), 0, 10)
et_preds = np.clip(et.predict(X_test), 0, 10)






from sklearn.metrics import mean_squared_error

dt_mse = mean_squared_error(y_test, dt_preds)
rf_mse = mean_squared_error(y_test, rf_preds)
gb_mse = mean_squared_error(y_test, gb_preds)
et_mse = mean_squared_error(y_test, et_preds)

print(dt_mse, rf_mse, gb_mse, et_mse)

summary = pd.DataFrame(
    {
        'DT predictions' : dt_preds,
        'RF predictions': rf_preds,
        'GB predictions': gb_preds,
        'ET predictions' : et_preds,
        'Ensemble of the four ' : (dt_preds + rf_preds + gb_preds + et_preds) / 4,
        'Real values' : y_test
    }
)
summary

summary.to_csv('./db/summary01.csv')
# # some vlaue is -0.0169168041206813 I could bounded the output






from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(max_features=4806,stop_words='english')
vector  = cv.fit_transform(anime['genre'].values.astype('U')).toarray()
similarity = cosine_similarity(vector)
similarity

anime['genre'].values.astype('U')

# Input the name of the anime you want to find similar ones for
input_anime = 'Naruto: Shippuuden'

# Check if the input anime exists in the 'anime' DataFrame
if input_anime not in anime['title'].values:
    print(f"{input_anime} not found in anime titles")
else:
    # Get the index of the input anime
    input_anime_index = anime[anime['title'] == input_anime].index[0]

    # Extract the cosine similarity scores for the input anime
    similarity_scores = list(enumerate(similarity[input_anime_index]))

    # Sort the cosine similarity scores in descending order
    sorted_scores = sorted(similarity_scores,key=lambda x:x[1],reverse=True)

    # Extract the top n similar anime
    n = 10
    top_n_anime_indices = [i[0] for i in sorted_scores[1:n+1]]

    # Return the names of the top n similar anime
    print(anime.iloc[top_n_anime_indices]['title'].values)


