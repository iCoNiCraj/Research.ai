from fastapi import FastAPI
from typing import Union
from tools import process_input
from llm import process_url

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/query")
def read_query(q: Union[str, None] = None):
    if q:
        result = process_input(q)
        return {"papers": result}
    return {"query": "No query provided"}

@app.get("/create_podcast")
def create_podcast(url: Union[str, None] = None):
    if url:
        script = process_url(url)
        file_path = "podcast.mp3"
        
    return {"query": "No query provided"}
