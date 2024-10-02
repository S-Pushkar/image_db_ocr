# image_db_ocr

## Description

Exposes an API to extract text from images using EasyOCR.

## Usage
```uvicorn main:app --reload```

## Sample cURL request
```curl -X POST -F "file=@/home/pushkars/Desktop/SomeProject/image_db_ocr/test_images/emotional_image.jpg" http://127.0.0.1:8000/imagetotext/```