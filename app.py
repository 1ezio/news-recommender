import pandas as pd
import numpy as np 
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from fastapi import FastAPI
from pydantic import BaseModel

class  recommendationClass(BaseModel):
    ministry:str

app = FastAPI()

@app.post("/")
async def getRecommendation(recommendation:recommendationClass):    
    count = CountVectorizer()
    textDataNotCleaned = pd.read_csv("ministries.csv")
    textData = pd.read_csv("ministriesArticles_lemmatized.csv")

    textDataNotCleaned = textDataNotCleaned.dropna()
    textData = textData.dropna()

    countMatrix = count.fit_transform(textData['text'])
    cosineSim = cosine_similarity(countMatrix, countMatrix)
    indices = pd.Series(textDataNotCleaned['target'])
        
    def recommend(title, cosine_sim = cosineSim):
        recommendedMovies = []
        idx = indices[indices == title].index[0]   
        scoreSeries = pd.Series(cosineSim[idx]).sort_values(ascending = False)
        topFiveRecommends = list(scoreSeries.iloc[1:6].index)   
        
        for i in topFiveRecommends:
            recommendedMovies.append(list(textDataNotCleaned['text'])[i])
            print(list(textData['target'])[i])
            
        return recommendedMovies
    return[{ "Recommedations": (recommend("Ministry of Tourism"))}]


