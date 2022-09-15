


import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
movie_df=pd.read_csv("movies.csv")
df=pd.read_csv("ratings.csv")
movie_df['genres']=movie_df['genres'].str.replace("|"," ")
movie_df['title']=movie_df['title'].str.replace('(\(\d\d\d\d\))','')
movie_df['title']=movie_df['title'].apply(lambda x:x.strip())
st.title("Find Movie :")
st.write(movie_df)
st.title("Find UserId :")
st.write(df)
option = st.selectbox(
     'Select Type of Recommender System',
     ('Popularity-Based Recommender System', 'Content-Based Recommender System', 'Collaborative Based Recommender System'))



st.title(option)



type_movies=movie_df.groupby("genres")["movieId"].sum().sort_values(ascending=False)

merged_left = pd.merge(left=movie_df, right=df, how='left', left_on='movieId', right_on='movieId')

def movie_recommend(original_title):
    idx = indices[original_title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return movie_title.iloc[movie_indices]

def recommendMovie(title,topN):
  result=pd.DataFrame(movie_recommend(title).head(topN))
  return result

if option=='Popularity-Based Recommender System':
     ge=st.text_input("Genre(g):","Comedy")
     th=st.text_input("Minimum reviews threshold(t):",100)
     re=st.text_input("Num recommendations (N) :",5)
     out=merged_left[merged_left["genres"]==ge ].sort_values(by=["genres","rating","userId"], ascending=False)
     out=out[out["userId"]>=int(th)]
     out["Num Reviews"]=out.userId.astype("int")
     out["Movie Title"]=out.title
     out["Average Movie Rating"]=out.rating.astype("float")
     out=out.reset_index(drop=True)
     final=out[["Movie Title","Average Movie Rating","Num Reviews"]]
     st.write(final.head(int(re)))
elif option=='Content-Based Recommender System':
    movie_df['genres'] = movie_df['genres'].str.replace("|", " ")
    movie_df['title'] = movie_df['title'].str.replace('(\(\d\d\d\d\))', '')
    movie_df['title'] = movie_df['title'].apply(lambda x: x.strip())


    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    matrix = tf.fit_transform(movie_df['genres'])

    cosine_similarities = linear_kernel(matrix,matrix)
    movie_title = movie_df['title']
    indices = pd.Series(movie_df.index, index=movie_df['title'])
    title=st.text_input("Movie Title (t): ",'Jumanji')
    topN=st.text_input("Num recommendations (N):",5)
    final=recommendMovie(title,int(topN))
    final = final.rename(columns={'title': 'Movie Title'})
    final.index.name='Movie Id'
    final.reset_index(level=0, inplace=True)
    st.write(final)
else:
    user = st.text_input("UserID:", 1)
    topN = st.text_input("Num recommendations(N):", 10)
    threshold = st.text_input("Threshold for similar users (k):", 100)
    data = pd.merge(movie_df, df, on="movieId", how="inner")
    user_movies_df = data.pivot_table(index='userId', columns='title', values='rating')
    matrix_norm = user_movies_df.subtract(user_movies_df.mean(axis=1), axis='rows')
    user_similarity = matrix_norm.T.corr()
    picked_userId =int(user)
    user_similarity.drop(index=picked_userId, inplace=True)
    n = int(threshold)
    user_similarity_threshold = 0.8
    similar_users = user_similarity[user_similarity[picked_userId] > user_similarity_threshold][picked_userId].sort_values(ascending=False)[:n]
    picked_userId_watched = matrix_norm[matrix_norm.index == picked_userId].dropna(axis=1, how='all')
    similar_user_movie = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1, how='all')
    similar_user_movie.drop(picked_userId_watched.columns, axis=1, inplace=True, errors='ignore')
    item_score = {}

    for i in similar_user_movie.columns:
        movie_rating = similar_user_movie[i]
        total = 0
        count = 0
        for u in similar_users.index:
            if pd.isna(movie_rating[u]) == False:
                score = similar_users[u] * movie_rating[u]
                total += score
                count += 1
        item_score[i] = total / count
    item_score = pd.DataFrame(item_score.keys(), columns=['Movie Title'])
    final=item_score.head(int(topN))
    st.write(final)