from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
import easyocr

app = FastAPI()
reader = easyocr.Reader(['en', 'kn'])

@app.get("/hello")
def read_root():
    return {"Hello": "World"}

def read_text_from_image(image):
    result = reader.readtext(image, detail=0)
    print(result)

@app.post("/uploadfile/")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    image = file.file.read()
    background_tasks.add_task(read_text_from_image, image)
    return JSONResponse(content={"message": "success"})