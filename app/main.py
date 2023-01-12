from typing import List, Optional
from fastapi import FastAPI, Query
from app.resolver import random_items, random_genre_items
from app.recommender import item_based_recommendation,user_based_recommendation
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/all/")
async def all_movies():
    result = random_items()
    return {"message": result}

@app.get("/genres/{genre}")
async def genre_movies(genre: str):
    result= random_genre_items(genre)
    return {"meassage" : result}

@app.get("/user-based/")
async def user_based(params: Optional[List[str]] = Query(None)):
    input_ratings_dict = dict(
        (int(x.split(":")[0]), float(x.split(":")[1])) for x in params
    )
    result=user_based_recommendation(input_ratings_dict)
    return {"message": result}

@app.get("/item-based/{item_id}")
async def item_based(item_id: str):
    result=item_based_recommendation(item_id)
    return {"message":result}