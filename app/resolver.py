import pandas as pd
item_fname = 'data/movies_final.csv'

def random_items():
    movies_df =pd.read_csv(item_fname)
    movies_df = movies_df.fillna('')
    result_items = movies_df.sample(10).to_dict(orient='records')
    return result_items

def random_genre_items(genre):
    movies_df =pd.read_csv(item_fname)
    genre_df = movies_df[movies_df['genres'].apply(lambda x: genre in x.lower())]
    result_items = genre_df.sample(10).to_dict(orient='records') 
    return result_items

