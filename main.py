from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body,Request,Path
from fastapi.responses import FileResponse,JSONResponse,StreamingResponse
from typing import List
import os
import json
import shutil
import aiofiles
import asyncio
from pydantic import BaseModel
from typing import List, Optional
from jose import JWTError, jwt,ExpiredSignatureError
from contextlib import asynccontextmanager
from datetime import datetime
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import tensorflow as tf
import keras
from keras import layers
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D,Dropout,Flatten
from keras import models,layers, regularizers
import numpy as np
import cv2
import tensorflow as tf
import io
from PIL import Image


# from sqlalchemy import Column, String, Float, DateTime, ForeignKey, create_engine, Boolean
# from sqlalchemy.orm import relationship, sessionmaker,Session,declarative_base
# from google.cloud.sql.connector import Connector

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

class TrainingParams(BaseModel):
    learning_rate: float = 0.001
    batch_size: int = 2
    epochs: int = 10
    steps_per_epoch: Optional[int] = None
    shuffle: bool = False
    do_fine_tuning: bool = False
    l1_regularizer: float = 0.0
    l2_regularizer: float = 0.0001
    label_smoothing: float = 0.1
    do_data_augmentation: bool = True
    decay_samples: int = 2560000
    warmup_epochs: int = 2


# create FastAPI APP
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    #when training completed,remove images
    if TRAINING_STATUS["current_user"] is not None and (TRAINING_STATUS["status"] == "failed" or TRAINING_STATUS["status"] == "completed"):
        await reset_system_internal(TRAINING_STATUS["current_user"])


app = FastAPI(lifespan=lifespan)


class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.headers.get("X-Forwarded-Proto", "http") == "https":
            request.scope["scheme"] = "https"
        response = await call_next(request)
        return response

app.add_middleware(HTTPSRedirectMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許的來源
    allow_credentials=False,
    allow_methods=["*"],  # 允許所有 HTTP 方法
    allow_headers=["*"],  # 允許所有標頭
)


MODEL_FILE_PATH = "exported_model_test/temp"  # 剛訓練好的模型檔案路徑
USER_MODEL_PATH="exported_model_test/storage" #使用者模型檔案路徑
UPLOAD_FOLDER = "uploads"
USER_TEMP_PATH="exported_heatmap/heatmap_temp" #temp folder for saving heatmap image

os.makedirs(MODEL_FILE_PATH, exist_ok=True)
os.makedirs(USER_MODEL_PATH, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 將 "static" 目錄掛載為靜態文件路徑
app.mount("/static", StaticFiles(directory=USER_MODEL_PATH), name="static-files")


# 訓練狀態idle:空閒, training:訓練中, completed:訓練完成, failed:訓練失敗
TRAINING_STATUS = {"status": "idle", "accuracy": None,"current_user": None}  
# 上傳狀態idle:空閒, uploading:上傳中, completed:上傳完成
UPLOADING_STATUS = {"status": "idle", "current_users": set() }  

@app.get("/")
async def main():
    return FileResponse('index.html')

@app.get("/modeltest/",include_in_schema=False)
async def modeltest():
    return FileResponse('modeltest.html')
@app.get("/modeltest/script.js",include_in_schema=False)
async def script():
    return FileResponse('script.js')
@app.get("/modeltest/style.css",include_in_schema=False)
async def style():
    return FileResponse('style.css')

@app.get("/modeltest/exported_model_test/{useremail}/model.tflite")
async def getmodel(useremail:str=Path(...)):
    return FileResponse(f'exported_model_test/{useremail}/model.tflite')


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

async def save_uploaded_file(file: UploadFile, category_name: str,user_email:str) -> str:
    # 建立圖片分類資料夾
    category_path = os.path.join(UPLOAD_FOLDER,user_email, category_name)
    os.makedirs(category_path, exist_ok=True)
    file_location = os.path.join(category_path, file.filename)

    # 異步寫入檔案
    async with aiofiles.open(file_location, "wb") as buffer:
        await buffer.write(await file.read())

    return file_location

#reset training task
async def reset_system_internal(user_email: str):
    global TRAINING_STATUS
    
    # 清除用戶上傳的圖片
    user_upload_path = os.path.join(UPLOAD_FOLDER, user_email)
    if os.path.exists(user_upload_path):
        shutil.rmtree(user_upload_path)
    
    # 重置訓練狀態
    TRAINING_STATUS = {"status": "idle", "accuracy": None, "current_user": None}
    print("System has been reset and is ready for new training.")
    

#clean model temp
async def reset_model_internal(user_email: str):
    model_path = os.path.join(MODEL_FILE_PATH,user_email)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
   
    return "temporary Model files have been removed."

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


@app.post("/uploadfiles/")
#上傳圖片，一定要有檔案和類別名稱
async def upload_files(request:Request, files: List[UploadFile] = File(...), category_name: str = Body(...)):
    global UPLOADING_STATUS
    user_email=tokenChecker(request)
    if user_email is None:
        raise HTTPException(status_code=422, detail="token is invalid,please login again")
    if category_name is None:
        raise HTTPException(status_code=422, detail="Category name cannot be empty")
    
    print('file number',len(files)) # 這裡會印出上傳的檔案數量
    UPLOADING_STATUS["status"] = "uploading"
    UPLOADING_STATUS["current_users"].add(user_email)

    for file in files:
        await save_uploaded_file(file, category_name,user_email)
    
    if len(UPLOADING_STATUS["current_users"])==0:
        UPLOADING_STATUS["status"] = "completed"
    
    UPLOADING_STATUS["current_users"].discard(user_email)
    
    return {"message": f"Files uploaded successfully to category: {category_name}"}

#get image category labels
@app.get("/labels/",description="get uploaded image category labels")
async def get_labels(request:Request):
    """get image category labels"""
    user_email=tokenChecker(request)
    if user_email is None:
        return {"labels":["error","token is invalid,please login again"]}
    
    user_path = os.path.join(UPLOAD_FOLDER,user_email)
    
    if os.path.exists(user_path):
        cat_files={}
        directory = [d for d in os.listdir(user_path) if os.path.isdir(os.path.join(user_path, d))]
        for d in directory:
            file_num= len([f for f in os.listdir(os.path.join(user_path, d))])
            cat_files[d]=file_num

        return {"labels": os.listdir(user_path),**cat_files}
    else:
        return {"labels": []}


@app.post("/train/")
async def train_model(background_tasks: BackgroundTasks,request:Request,TaskName:str=Body(...), training_params: TrainingParams=Body(...)):
    
    global TRAINING_STATUS
    # get image category labels
    labels = await get_labels(request)
    if labels.get("labels") == []:
        return {"message": "No images uploaded for training."}
    elif len(labels.get("labels"))<2:
        return {"message": "More than two categories are necessary."}
    
    user_email=tokenChecker(request)
    
    usertask=await get_user_tasks(request)
    if TaskName in usertask["user_tasks"]:
        return {"message": "Task name already exists. Please choose a different name."}

    if TRAINING_STATUS["current_user"] == user_email:
        if TRAINING_STATUS["status"] == "training":
            return {"message": "Model is already training."}
        elif TRAINING_STATUS["status"] == "completed":
            return {"message": "Model has already been trained."}
    elif TRAINING_STATUS["current_user"] != user_email:
        if TRAINING_STATUS["status"] == "training":
            return {"message": "Another user is currently training a model. Please try again later."}
        elif TRAINING_STATUS["status"] == "completed":
            return {"message": "Previous training has completed. Please wait for system reset."}
        elif TRAINING_STATUS["status"] == "failed":
            return {"message": "Previous training has failed. Please wait for system reset."}
    
    TRAINING_STATUS["status"] = "training"
    TRAINING_STATUS["current_user"] = user_email

    await reset_model_internal(user_email)
    background_tasks.add_task(train_model_task,user_email,TaskName,training_params)

    return {"message": "Model training started in the background"}


async def train_model_task(user_email:str,TaskName:str,training_params: TrainingParams):
    global TRAINING_STATUS
    
    try:
        
        image_path = os.path.join(UPLOAD_FOLDER,user_email)
        #error code
        if os.path.exists(image_path) !=True:
            raise Exception
        
        

        # 設置數據集目錄
        dataset_dir = image_path
        category_label=sorted([d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path,d))])
        category_number=len(category_label)
        
        # 定義批次大小和圖像尺寸
        batch_size = training_params.batch_size
        img_size = (224, 224)

        # 設置隨機種子以確保可重現性
        seed = 123

        # 創建訓練數據集（70%）
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.3,  # 先將 30% 的數據拆分出來
            subset="training",
            seed=seed,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical"
        )

        # 創建驗證數據集（15%）
        val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.3,  # 使用相同的 30% 數據
            subset="validation",
            seed=seed,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical"
        )

        # 從驗證數據集中再拆分出測試數據集（15%）
        val_batches = tf.data.experimental.cardinality(val_dataset)
        test_dataset = val_dataset.take(val_batches // 2)
        val_dataset = val_dataset.skip(val_batches // 2)

        # 檢查數據集大小
        print(f'Training batches: {tf.data.experimental.cardinality(train_dataset)}')
        print(f'Validation batches: {tf.data.experimental.cardinality(val_dataset)}')
        print(f'Test batches: {tf.data.experimental.cardinality(test_dataset)}')
            
       
        


        # 定義數據增強層
        augmentation_layers = [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]

        # 定義數據增強函數
        def data_augmentation(x):
            for layer in augmentation_layers:
                x = layer(x)  # 每個數據增強層都是可調用的
            return x
        def preprocess_v2(x,y):
            x=keras.applications.mobilenet_v2.preprocess_input(x)
            return x,y
        # augmentation and MobileNetV2 preprocess
        def augment_and_preprocess(x, y):
            x = data_augmentation(x)
            x,y = preprocess_v2(x,y)  # 將增強後的圖像進行預處理
            return x, y

        # 對訓練數據集應用數據增強
        train_dataset = train_dataset.map(augment_and_preprocess)
        test_dataset=test_dataset.map(preprocess_v2)
        val_dataset=val_dataset.map(preprocess_v2)
       

        # 顯示數據集的一些增強後的圖像


       


        base_model = MobileNetV2(input_shape=(224, 224, 3),
                                    include_top=False,
                                    weights='imagenet')
        # Do not include the ImageNet classifier at the top.

        # Freeze the base_model
        base_model.trainable = False
        if training_params.do_fine_tuning:
            base_model.trainable = True

        
        # Pre-trained Xception and MobileNetV2 weights requires that input be scaled
        # from (0, 255) to a range of (-1., +1.), the rescaling layer
        # outputs: `(inputs * scale) + offset`
        # scale_layer = keras.layers.Rescaling(scale=1./127.5,offset=-1)
        # x = scale_layer(inputs)

        # The base model contains batchnorm layers. We want to keep them in inference mode
        # when we unfreeze the base model for fine-tuning, so we make sure that the
        # base_model is running in inference mode here.
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Flatten(name="flatten")(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.4)(x)  # Regularize with dropout
        # Output layer and loss function based on the number of categories
        if category_number == 2:
            outputs = layers.Dense(2, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=training_params.l1_regularizer, l2=training_params.l2_regularizer))(x)
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        else:
            outputs = layers.Dense(category_number, activation='softmax', kernel_regularizer=regularizers.l1_l2(l1=training_params.l1_regularizer, l2=training_params.l2_regularizer))(x)
            loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=training_params.label_smoothing)

        model = keras.Model(base_model.input, outputs)


        model.summary(show_trainable=True)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=training_params.learning_rate),
              loss=loss_fn,
              metrics=['accuracy'])
        
        import time
        startime = time.time()
        epochs =training_params.epochs

        print("Fitting the top layer of the model")

        await asyncio.to_thread(model.fit,train_dataset, epochs=epochs, validation_data=val_dataset)
        
        endtime = time.time()
        print("Time taken: ", endtime - startime)
        model_save_path = os.path.join(MODEL_FILE_PATH,user_email)
        print(model_save_path)
        rawmodel_path = os.path.join(model_save_path, 'temp_model')
        model.save(rawmodel_path)

        # load model from SavedModel 
        converter = tf.lite.TFLiteConverter.from_saved_model(rawmodel_path)
        tflite_model = converter.convert()
        # model file temp path
        tflite_model_path = os.path.join(model_save_path,'model.tflite')
        # convert to .tflite
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        
        labels_dict = {"label": category_label}
       
        # output label json ,temp path
        model_labels_path = os.path.join(model_save_path,'model_labels.json')
        with open(model_labels_path,'w', encoding='utf-8') as json_file:
            json.dump(labels_dict, json_file, ensure_ascii=False, indent=4)

        print("[model_labels.json] exported")
        
        loss, accuracy = model.evaluate(test_dataset)
        
        print(f"Loss: {loss}, Accuracy: {accuracy}---{user_email}")
        TRAINING_STATUS["accuracy"] = accuracy
        TRAINING_STATUS["status"] = "completed"
        
        print(TRAINING_STATUS)

        #copy model.tflite( in model temp) labels.json  to user model path and rename model.tflite to datetime+taskname.tflite


        user_model_path = os.path.join(USER_MODEL_PATH,user_email)
        os.makedirs(user_model_path, exist_ok=True)
        model_name=f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{TaskName}.tflite"
        label_name=f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{TaskName}.json"
        rawmodel_name=f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{TaskName}"
        shutil.copy(tflite_model_path,os.path.join(user_model_path,model_name))
        shutil.copy(model_labels_path,os.path.join(user_model_path,label_name))
        shutil.copytree(rawmodel_path, os.path.join(user_model_path,rawmodel_name), dirs_exist_ok=True)
        print(f"Model copied to: {user_model_path}")
        #delay 1 minute to remove temp model file
        await asyncio.sleep(60)
        await reset_system_internal(user_email)

    except Exception as e:
        TRAINING_STATUS["status"] = "failed"
        TRAINING_STATUS["accuracy"] = None
        print(f"Training failed: {e}")

 
        

@app.get("/reset_system/")
async def reset_system(request: Request):

    global TRAINING_STATUS
    user_email = tokenChecker(request)
    if user_email is None:
        raise HTTPException(status_code=422, detail="token is invalid,please login again")
    
    await reset_system_internal(user_email)
    return {"message": "System has been reset and is ready for new training."}





@app.get("/training_status/")
async def get_training_status():
    return TRAINING_STATUS

@app.get("/uploading_status/",description="get uploading status")
async def get_uploading_status(request:Request):
    global UPLOADING_STATUS
    user_email=tokenChecker(request)
    
    if user_email is None:
        raise HTTPException(status_code=422, detail="token is invalid,please login again")
    
    if user_email not in UPLOADING_STATUS["current_users"]:
        return {"status": "stopped"}
    return {"status": "uploading"}

@app.get("/users/me")
async def read_users_me(request: Request)->str:
    user_email = tokenChecker(request)
    if user_email is None:
        raise HTTPException(status_code=422, detail="token is invalid,please login again")
    return user_email
    

#download model

@app.get("/download_model/{taskName}")
async def download_model(request: Request, taskName: str):
    user_email = tokenChecker(request)
    if user_email is None:
        raise HTTPException(status_code=422, detail="token is invalid,please login again")
    
    user_model_dir = os.path.join(USER_MODEL_PATH, user_email)
    
    if not os.path.exists(user_model_dir):
        raise HTTPException(status_code=404, detail="User model directory not found.")
    
    # 搜索匹配的模型文件
    matching_models = [f for f in os.listdir(user_model_dir) if f.endswith(f"-{taskName}.tflite")]
    matching_jsons = [f for f in os.listdir(user_model_dir) if f.endswith(f"-{taskName}.json")]
    
    if not matching_models:
        raise HTTPException(status_code=404, detail=f"No model found for task: {taskName}")
    
    # select newest model
    latest_model = max(matching_models)
    latest_json = max(matching_jsons)
    
    model_path = os.path.join(user_model_dir, latest_model)
    
    with open(os.path.join(user_model_dir,latest_json),"r",encoding="utf-8") as jsonfile:
        label_data=json.load(jsonfile)
    


    # 生成相對於靜態文件目錄的相對路徑
    relative_path = os.path.relpath(model_path, start="exported_model_test/storage")
    
    # 構建完整的URL
    model_url = request.url_for("static-files", path=relative_path)
    # if "http://" in model_url:
    #     model_url=model_url.replace("http://", "https://")
    return JSONResponse({
        "task_name": taskName,
        "model_file_url": str(model_url),
        "model_file_name": latest_model,
        "model_label":label_data["label"]
    })   
    
#delete model
@app.delete("/delete_model/{taskName}")
async def delete_model(request: Request, taskName: str):
    user_email = tokenChecker(request)
    if user_email is None:
        raise HTTPException(status_code=422, detail="token is invalid,please login again")
    
    user_model_dir = os.path.join(USER_MODEL_PATH, user_email)
    
    if not os.path.exists(user_model_dir):
        raise HTTPException(status_code=404, detail="User model directory not found.")
    
    # search model files
    matching_models = [f for f in os.listdir(user_model_dir) if f.endswith(f"-{taskName}.tflite")]
    matching_jsons = [f for f in os.listdir(user_model_dir) if f.endswith(f"-{taskName}.json")]
    matching_dirs = [d for d in os.listdir(user_model_dir) if d.endswith(f"-{taskName}") and os.path.isdir(os.path.join(user_model_dir, d))]
    
    # delete tflite file
    for model in matching_models:
        model_path = os.path.join(user_model_dir, model)
        os.remove(model_path)
    
    # delete json
    for json_file in matching_jsons:
        json_path = os.path.join(user_model_dir, json_file)
        os.remove(json_path)
    
    # delete raw model
    for dir in matching_dirs:
        dir_path = os.path.join(user_model_dir, dir)
        shutil.rmtree(dir_path)
    
    # 確認至少有一個項目被刪除
    if not (matching_models or matching_jsons or matching_dirs):
        raise HTTPException(status_code=404, detail=f"No model, JSON, or directory found for task: {taskName}")

    return {"message": f"tensorflow lite Model(s), label JSON file(s), and raw model directory(ies) for task: [{taskName}] deleted successfully."}

@app.get("/user_tasks/")
async def get_user_tasks(request: Request):
    user_email = tokenChecker(request)
    if user_email is None:
        raise HTTPException(status_code=422, detail="token is invalid,please login again")
    user_model_dir = os.path.join(USER_MODEL_PATH, user_email)
    
    if not os.path.exists(user_model_dir):
        return {"user_tasks": []}
    
    # 搜索所有模型文件
    models = [f.split('-')[1].split('.')[0] for f in os.listdir(user_model_dir) if f.endswith(".tflite")]

    return {"user_tasks": models}

#grad Cam
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
            gradcam_image = plot_and_combine_heatmap(heatmap,img2,predicted_class,probabilities,label_data["label"],os.path.join(USER_TEMP_PATH,user_email,"heatmap.jpg"))  # Pass filename for saving
           
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
