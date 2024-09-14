from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import easyocr
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()
reader = easyocr.Reader(['en', 'kn'])
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.get("/hello")
def read_root():
    return {"Hello": "World"}

def read_text_from_image(image):
    result = reader.readtext(image, detail=0)
    print(result)
    if len(result) > 0:
        vectors = model.encode(result)
        combined_vector = np.mean(vectors, axis=0)
        print(vectors)
        print(combined_vector)

@app.post("/uploadfile/")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    image = file.file.read()
    background_tasks.add_task(read_text_from_image, image)
    return {"message": "success"}