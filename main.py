from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import easyocr

app = FastAPI()
reader = easyocr.Reader(['en', 'kn'])

@app.get("/hello")
def read_root():
    return {"Hello": "World"}


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    image = file.file.read()
    result = reader.readtext(image, detail=0)
    return JSONResponse(content={"result": result})