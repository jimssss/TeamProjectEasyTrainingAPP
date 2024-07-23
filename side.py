from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body,Request,Path
from fastapi.responses import FileResponse,JSONResponse,StreamingResponse
import os
import json
import shutil
import aiofiles
import asyncio
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
from jose import JWTError, jwt,ExpiredSignatureError
from contextlib import asynccontextmanager
from datetime import datetime
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
import tensorflow as tf
import keras
from keras import layers
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D,Dropout,Flatten
from keras import models,layers, regularizers
from google.cloud import storage
from google.oauth2 import service_account
import datetime
import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
import io
from PIL import Image

USER_TEMP_PATH="exported_heatmap/heatmap_temp" #temp folder for saving heatmap image
USER_MODEL_PATH="exported_model_test/storage" 

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
BUCKET_NAME =os.getenv("BUCKET_NAME")

credentials_path = "/home/jimssss/jimDev2/temp/jimkey.json"
    
if not credentials_path:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS error")

credentials = service_account.Credentials.from_service_account_file(
    credentials_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

app = FastAPI()

# class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
#         if request.headers.get("X-Forwarded-Proto", "http") == "https":
#             request.scope["scheme"] = "https"
#         response = await call_next(request)
#         return response

# app.add_middleware(HTTPSRedirectMiddleware)

#check token 
def tokenChecker(request:Request)->str|None:

    token = request.headers.get("Authorization")
    print(token)
    try:
        print("decoding-------------------------------------")
        token=process_bearer_token(token)
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print(payload)
        user_email= payload.get("sub")
        if user_email is None:
            raise HTTPException(
                status_code=401,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"}
            )    
        return user_email
    
    except ExpiredSignatureError:
       print("ExpiredSignature")
       raise HTTPException(
                status_code=401,
                detail="ExpiredSignature",
                headers={"WWW-Authenticate": "Bearer"}
            )   
        
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
#check "bearer" 
def process_bearer_token(authorization: str) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is missing")
    
    parts = authorization.split()
    
    if parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header. Must start with 'Bearer'")
    elif len(parts) == 1:
        raise HTTPException(status_code=401, detail="Token missing")
    elif len(parts) > 2:
        raise HTTPException(status_code=401, detail="Invalid authorization header. Token contains spaces")

    token = parts[1].strip()
    return token

async def download_rawmodel(taskName:str,blob_path:str,temp_folder:str)->Optional[Tuple[str, Dict]]:
    """Download raw model files from GCS return temp folder path and label data"""
    
    # use the service account credentials to create a client
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(BUCKET_NAME)
    print(f"Searching for {taskName} raw model files in {BUCKET_NAME}/{blob_path}...")
    try:
        blobs = bucket.list_blobs(prefix=blob_path)
        # copy rawmodel directory to temp folder
        for blob in blobs:
            
            if (f'-{taskName}/') in blob.name: 
                folder_name = taskName 
                temp_path = f"{temp_folder}{folder_name}/"
                
                os.makedirs(temp_path, exist_ok=True)
                relative_path = blob.name.split(f'-{taskName}/')[1]
                if "/" in relative_path:
                    # create the directory if it doesn't exist
                    os.makedirs(f"{temp_path}{relative_path.split('/')[0]}", exist_ok=True)
            
                file_path = f"{temp_path}{blob.name.split(f'-{taskName}/')[1]}"
                
                # download the file to the temp folder
                blob.download_to_filename(file_path) 
                print(f"Downloaded: {file_path}")


            elif blob.name.endswith(f"-{taskName}.json"):
                    # load the JSON content from the blob
                    content = blob.download_as_bytes()
                    try:
                        json_data = json.loads(content.decode('utf-8'))
                        model_labeldata=json_data
                        print(f"  Loaded JSON content from {blob.name}")
                    except json.JSONDecodeError:
                        print(f"  Error: Unable to parse JSON content from {blob.name}")

        if temp_path and model_labeldata:
            return temp_path,model_labeldata
        else:
            return None,None
    except Exception as e:
        print(f"搜索模型文件時發生錯誤: {str(e)}")
        return None,None

async def search_model(taskName:str,blob_path:str)->Optional[Tuple[str, Dict]]:
    
    
    # use the service account credentials to create a client
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(BUCKET_NAME)
    print(f"Searching for {taskName} model files in {BUCKET_NAME}/{blob_path}...")
    try:
        blobs = bucket.list_blobs(prefix=blob_path)
        model_Blobname=""
        model_labeldata={}
        for blob in blobs:
            # remove the path prefix to get the relative path
            relative_path = blob.name[len(blob_path):]
            # check if the blob is a file and the file name matches
            if (not blob.name.endswith('/')) and ('/' not in relative_path):
                if relative_path.endswith(f"-{taskName}.tflite"):
                    model_Blobname=blob.name
                    print(f"Found model file: {model_Blobname}")
                elif relative_path.endswith(f"-{taskName}.json"):
                    # load the JSON content from the blob
                    content = blob.download_as_bytes()
                    try:
                        json_data = json.loads(content.decode('utf-8'))
                        model_labeldata=json_data
                        print(f"  Loaded JSON content from {blob.name}")
                    except json.JSONDecodeError:
                        print(f"  Error: Unable to parse JSON content from {blob.name}")
        
        if model_Blobname and model_labeldata:
            return model_Blobname,model_labeldata
        else:
            return None,None
    except Exception as e:
        print(f"搜索模型文件時發生錯誤: {str(e)}")
        return None,None



def generate_signed_url(bucket_name, blob_name, expiration):
    """create a signed URL for a blob in a bucket"""
   
    
    # use the service account credentials to create a client
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
   

    url = blob.generate_signed_url(
        version="v4",
        # 这个 URL 将在 'expiration' 秒后过期
        expiration=datetime.timedelta(seconds=expiration),
        # 允许 GET 请求使用这个 URL
        method="GET",
    )

    return url



def gradcam(model, image, last_conv_layer_name):
    """return heatmap,predicted_class,probabilities"""
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(image)
        class_out = model_out[:, np.argmax(model_out[0])]
        predicted_class = np.argmax(model_out[0])
        print(f"Predicted class: {predicted_class}",model_out[0])
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

   
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap_shape = (grads.shape[1], grads.shape[2])
    heatmap_Emphasis = np.maximum(heatmap, 0) # ReLU
    heatmap_Emphasis /= np.max(heatmap_Emphasis) # normalization
    heatmap_Emphasis = heatmap_Emphasis.reshape(heatmap_shape)
    return heatmap_Emphasis,predicted_class,model_out[0]


def plot_and_combine_heatmap(heatmap, img, predicted_class, probabilities,labeldata, output_path="heatmap.jpg"):
   
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # 添加预测结果和概率到图像上
    predicted_label = labeldata[predicted_class]
    probability = probabilities[predicted_class] * 100
    text = f"Prediction: {predicted_label}\nProbability: {probability:.2f}%"
    cv2.putText(superimposed_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imwrite(output_path, superimposed_img)
    return Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))

@app.get("/")
async def main():
    return FileResponse('side.html')

# download model api
@app.get("/download_model/{taskName}")
async def download_model(request: Request, taskName: str):
    user_email = tokenChecker(request)
    if user_email is None:
        raise HTTPException(status_code=422, detail="token is invalid,please login again")
    
    blob_path= f"storage/{user_email}/"
    # list all files in the directory
    blob_name,label_data = search_model(taskName,blob_path)
    print(f"blob_name:{blob_name}")
    if not blob_name:
        raise HTTPException(status_code=404, detail="User model not found.")
    
    model_url=generate_signed_url(BUCKET_NAME, blob_name, 3600)

    
    return {
        "task_name": taskName,
        "model_file_url": str(model_url),
        "model_file_name": blob_name[len(blob_path):],
        "model_label":label_data["label"]
    }   

# api for tensorflow lite model inference
# return prediction and gradcam image 
@app.post("/predict/{taskName}",include_in_schema=False)
async def predict(request: Request, taskName: str, file: UploadFile = File(...)):
    import keras.utils as ku
    user_email = tokenChecker(request)
    if user_email is None:
        raise HTTPException(status_code=422, detail="token is invalid,please login again")
    
    # Load the model
    # model_path,label_data=await download_rawmodel(taskName,f"storage/{user_email}/",f"{USER_TEMP_PATH}/{user_email}/")
    model_rootpath=f"{USER_MODEL_PATH}/{user_email}/"
    model_path=f"{USER_MODEL_PATH}/{user_email}/"
    
    for f in os.listdir(model_rootpath):
        # if isdir return True
        if os.path.isdir(f"{model_rootpath}{f}"):
            if f.endswith(f"-{taskName}"):
                model_path=os.path.join(model_rootpath,f)
                print(f"model_path:{model_path}")
        if f.endswith(f"-{taskName}.json"):
            with open(f"{model_rootpath}{f}","r") as j:
                label_data=json.load(j)
            
    
    print(f"model_path:{model_path}")
    print(f"label_data:{label_data}")
    if not model_path:
        raise HTTPException(status_code=404, detail="User model not found.")
    model = tf.keras.models.load_model(model_path)
    model.summary()
    

    try:
        # Read and preprocess the image
        contents = await file.read()
        image_stream = io.BytesIO(contents)
        img = Image.open(image_stream)
        # Convert PIL Image to OpenCV Image (RGB -> BGR for OpenCV)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        "backup image"
        img2=img.copy()
        # Resize image to 224x224
        img = cv2.resize(img, (224, 224))
        
        # Convert BGR image back to RGB for model prediction
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to array and preprocess
        img_array = ku.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Find the last convolutional layer
        conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        last_conv_layer_name = conv_layers[-1] if conv_layers else None

        print("Model convolutional layers:", conv_layers)
        print("Last convolutional layer name:", last_conv_layer_name)
        
        if last_conv_layer_name:
            heatmap,predicted_class,probabilities = gradcam(model, img_array, last_conv_layer_name)
            gradcam_image = plot_and_combine_heatmap(heatmap,img2,predicted_class,probabilities,label_data["label"],os.path.join(USER_TEMP_PATH,user_email))  # Pass filename for saving
           
            # Return the gradcam image, streaming it as PNG
            buf = io.BytesIO()
            gradcam_image.save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")
        else:
            raise HTTPException(status_code=500, detail="No convolutional layers found in model.")
        
    finally:
        await file.close()  # Ensure the file is closed

    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)