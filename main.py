from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from pydantic import EmailStr
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
from pymilvus import connections, utility
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
import easyocr
import time
import os
import json
from contextlib import asynccontextmanager
from kafka import KafkaConsumer
import threading
import base64
import cv2

def kafka_consumer_task():
    consumer = KafkaConsumer('upload-file', 
                             bootstrap_servers='localhost:9092',
                             auto_offset_reset='earliest',
                             enable_auto_commit=True,
                             group_id='upload-file-group',
                             value_deserializer=lambda x: json.loads(x.decode('utf-8')))
    for message in consumer:
        process_kafka_message(message.value)

def process_kafka_message(message: dict):
    email = message['email']
    image_path = message['fileName']
    image = message['fileContent']
    image = base64.b64decode(image)
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    read_text_from_image(image, email, image_path)

@asynccontextmanager
async def kafka_starter(app: FastAPI):
    thread = threading.Thread(target=kafka_consumer_task)
    thread.start()
    yield
    thread.join()

app = FastAPI(lifespan=kafka_starter)
reader = easyocr.Reader(['en', 'kn'])
model = SentenceTransformer('all-MiniLM-L6-v2')

load_dotenv(".env.local")
ZILLIZ_ENDPOINT = os.getenv("ZILLIZ_ENDPOINT")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")
connections.connect(uri=ZILLIZ_ENDPOINT, token=ZILLIZ_API_KEY)

COLLECTION_NAME = "ocr_text_vectors"

@app.get("/hello")
def hello():
    return {"Hello": "World"}

@app.post("/imagetotext/")
async def image_to_text(file: UploadFile = File(...)):
    image = file.file.read()
    result = reader.readtext(image, detail=0)
    return {"text": " ".join(result)}

def create_collection(collection_name):
    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True, description="The unique identifier for the document"),

            FieldSchema(name="email", dtype=DataType.VARCHAR, max_length=100, description="User email"),

            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500, description="Path to the image"),

            FieldSchema(name="timestamp", dtype=DataType.INT64, description="The timestamp of the document insertion"),

            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384, description="Text embedding vector represrnting the text in image")
        ]

        schema = CollectionSchema(fields=fields, description="OCR text vectors with image paths and user emails")
        
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        
        collection.create_index(field_name="embedding", index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"})
        
        return collection
    else:
        return Collection(name=COLLECTION_NAME)


def read_text_from_image(image, email, image_path):
    result = reader.readtext(image, detail=0)

    collection = create_collection(COLLECTION_NAME)

    if len(result) > 0:
        result = " ".join(result)
        vectors = model.encode(result)

        entities = [
            [email],
            [image_path],
            [int(time.time())],
            [vectors.tolist()]
        ]

        collection.insert(entities)
    else:
        entities = [
            [email],
            [image_path],
            [int(time.time())],
            [np.zeros(384)]
        ]

        collection.insert(entities)

    collection.flush()


@app.post("/uploadfile/")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), email: EmailStr = Form(...), image_path: str = Form(...)):
    image = file.file.read()
    background_tasks.add_task(read_text_from_image, image, email, image_path)
    return {"message": "success"}