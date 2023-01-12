import sys

import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from settings import api_key as ak
import pandas as pd
import requests
from tqdm import tqdm
import time
ak=ak.keychain()
def add_rating(df):
    ratings_df=pd.read_csv('app/data/ratings.csv')
    ratings_df['movieId']=ratings_df['movieId'].astype(str)
    agg_df=ratings_df.groupby('movieId').agg(
        rating_count=('rating','count'),
        rating_avg=('rating', 'mean')
        ).reset_index()
    rating_added_df=df.merge(agg_df,on='movieId')
    return rating_added_df

def add_poster(df):
    for i, row in tqdm(df.iterrows(),total=df.shape[0]):
        tmdb_id=row['tmdbId']
        url=f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={ak.TMDB_API_KEY}&language=en-US"
        result= requests.get(url)
        try:
            df.at[i,'poster_path']="https://image.tmdb.org/t/p/w500"+result.json()['poster_path']
            time.sleep(0.1)
        except(TypeError,KeyError) as e:
            df.at[i,'poster_path']=None
    return df

if __name__ == "__main__":
    movies_df=pd.read_csv('app/data/movies.csv')
    movies_df['movieId']=movies_df['movieId'].astype(str)
    links_df=pd.read_csv('app/data/links.csv',dtype=str)
    merged_df=pd.merge(movies_df,links_df,on='movieId',how='left')
    merged_df['url']=merged_df['imdbId'].apply(lambda x: f"https://www.imdb.com/title/tt{x}")
    result=add_rating(merged_df)
    result=add_poster(result)
    
    result.to_csv('app/data/movies_final.csv',index=None)