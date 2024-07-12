from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body,Request,Path
from fastapi.responses import FileResponse,JSONResponse
from typing import List
import os
import shutil
import aiofiles
import asyncio
import tensorflow as tf
from mediapipe_model_maker import image_classifier
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision
from pydantic import BaseModel
from typing import List, Optional
from jose import JWTError, jwt,ExpiredSignatureError
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

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


MODEL_FILE_PATH = "exported_model_test/temp"  # 剛訓練好的模型檔案路徑
USER_MODEL_PATH="exported_model_test/storage" #使用者模型檔案路徑
UPLOAD_FOLDER = "uploads"

os.makedirs(MODEL_FILE_PATH, exist_ok=True)
os.makedirs(USER_MODEL_PATH, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 將 "static" 目錄掛載為靜態文件路徑
app.mount("/static", StaticFiles(directory=USER_MODEL_PATH), name="static-files")


# 訓練狀態idle:空閒, training:訓練中, completed:訓練完成, failed:訓練失敗
TRAINING_STATUS = {"status": "idle", "accuracy": None,"current_user": None}  
# 上傳狀態idle:空閒, uploading:上傳中, completed:上傳完成
UPLOADING_STATUS = {"status": "idle", "current_users": set() }  


# GCP MySQL instance connection settings
INSTANCE_CONNECTION_NAME =os.getenv("INSTANCE_CONNECTION_NAME")
print(f"Your instance connection name is: {INSTANCE_CONNECTION_NAME}")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = "userdata"


#uncomment the following code to use database
# # initialize the connector object
# connector = Connector()

# # define a function that returns a database connection object
# def getconn():
#     conn = connector.connect(
#         INSTANCE_CONNECTION_NAME,
#         "pymysql",
#         user=DB_USER,
#         password=DB_PASS,
#         db=DB_NAME
#     )
#     return conn

# # create a connection pool
# pool = create_engine(
#     "mysql+pymysql://",
#     creator=getconn,
# )

# # create a database model
# Base = declarative_base()

# class User_Db(Base):
#     __tablename__ = "users"
#     email = Column(String(255), primary_key=True, unique=True, index=True, nullable=False)
#     hashed_password = Column(String(255), nullable=False)
#     line_name =Column(String(255),nullable=True)

# # class Task(Base):
# #     __tablename__ = "tasks"
# #     id = Column(String(36), primary_key=True, index=True)
# #     name = Column(String(36), index=True)
# #     user_id = Column(String(36), ForeignKey('users2.id'))
# #     file_path = Column(String(100))
# #     accuracy = Column(Float)
# #     created_at = Column(DateTime, default=datetime.datetime.utcnow)
# #     updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

# #     owner = relationship("User", back_populates="tasks")

# # create the database table
# Base.metadata.create_all(bind=pool)

# # create a session local to perform database operations
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=pool)

# # search user data
# def search_user_data(email) -> dict | None:
#     session = SessionLocal()
#     try:
#         queried_user = session.query(User_Db).filter(User_Db.email == email).first()
#         if queried_user:
#             return {
#                 "email": queried_user.email,
#                 "hashed_password": queried_user.hashed_password,
#                 "line_name": queried_user.line_name
#             }
#         return None
#     finally:
#         session.close()

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
    
    if TRAINING_STATUS["current_user"] == user_email:
        # 清除用戶上傳的圖片
        user_upload_path = os.path.join(UPLOAD_FOLDER, user_email)
        if os.path.exists(user_upload_path):
            shutil.rmtree(user_upload_path)
        
        # 重置訓練狀態
        TRAINING_STATUS = {"status": "idle", "accuracy": None, "current_user": None}
        print("System has been reset and is ready for new training.")
    else:
        return "Unable to reset system: user mismatch."

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
    
    user_email=tokenChecker(request)
    print(user_email)
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
        data = image_classifier.Dataset.from_folder(image_path)
        train_data, remaining_data = data.split(0.8)
        test_data, validation_data = remaining_data.split(0.5)
        spec = image_classifier.SupportedModels.MOBILENET_V2
        hparams = image_classifier.HParams(learning_rate=training_params.learning_rate,
                                           batch_size=training_params.batch_size,
                                           epochs=training_params.epochs,
                                           export_dir=os.path.join(MODEL_FILE_PATH,user_email))  # model export directory
        options = image_classifier.ImageClassifierOptions(supported_model=spec, hparams=hparams)
        model = await asyncio.to_thread(image_classifier.ImageClassifier.create,
                                        train_data=train_data,
                                        validation_data=validation_data,
                                        options=options)
        
        loss, accuracy = model.evaluate(test_data)
        print(f"Loss: {loss}, Accuracy: {accuracy}---{user_email}")
        TRAINING_STATUS["accuracy"] = accuracy
        TRAINING_STATUS["status"] = "completed"
        
        model.export_model()
        print(TRAINING_STATUS)
        
        #copy model.tflite( in model temp)  to user model path and rename model.tflite to datetime+taskname.tflite
        temp_model_file_path = os.path.join(MODEL_FILE_PATH,user_email,"model.tflite")
        user_model_path = os.path.join(USER_MODEL_PATH,user_email)
        os.makedirs(user_model_path, exist_ok=True)
        model_name=f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{TaskName}.tflite"
        shutil.copy(temp_model_file_path,os.path.join(user_model_path,model_name))
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
    
    await reset_system_internal(user_email)
    return {"message": "System has been reset and is ready for new training."}





@app.get("/training_status/")
async def get_training_status():
    return TRAINING_STATUS

@app.get("/uploading_status/",description="get uploading status")
async def get_uploading_status(request:Request):
    global UPLOADING_STATUS
    user_email=tokenChecker(request)
    if user_email not in UPLOADING_STATUS["current_users"]:
        return {"status": "stopped"}
    return {"status": "uploading"}

@app.get("/users/me")
async def read_users_me(request: Request)->str:
    user_email = tokenChecker(request)
    return user_email
    

#download model

@app.get("/download_model/{taskName}")
async def download_model(request: Request, taskName: str):
    user_email = tokenChecker(request)
    user_model_dir = os.path.join(USER_MODEL_PATH, user_email)
    
    if not os.path.exists(user_model_dir):
        raise HTTPException(status_code=404, detail="User model directory not found.")
    
    # 搜索匹配的模型文件
    matching_models = [f for f in os.listdir(user_model_dir) if f.endswith(f"-{taskName}.tflite")]
    
    if not matching_models:
        raise HTTPException(status_code=404, detail=f"No model found for task: {taskName}")
    
    # 選擇最新的模型（如果有多個）
    latest_model = max(matching_models)
    model_path = os.path.join(user_model_dir, latest_model)
    
    # 生成相對於靜態文件目錄的路徑
    relative_path = os.path.relpath(model_path, start="exported_model_test/storage")
    
    # 構建完整的URL
    model_url = request.url_for("static-files", path=relative_path)
    
    return JSONResponse({
        "task_name": taskName,
        "model_file_url": str(model_url),
        "model_file_name": latest_model
    })   
    
#delete model
@app.delete("/delete_model/{taskName}")
async def delete_model(request: Request, taskName: str):
    user_email = tokenChecker(request)
    user_model_dir = os.path.join(USER_MODEL_PATH, user_email)
    
    if not os.path.exists(user_model_dir):
        raise HTTPException(status_code=404, detail="User model directory not found.")
    
    # 搜索匹配的模型文件
    matching_models = [f for f in os.listdir(user_model_dir)if f.endswith(f"-{taskName}.tflite")] 
    if not matching_models:
        raise HTTPException(status_code=404, detail=f"No model found for task: {taskName}")
    
    # 刪除所有匹配的模型文件
    for model in matching_models:
        model_path = os.path.join(user_model_dir, model)
        os.remove(model_path)
    
    return {"message": f"Model(s) for task:[ {taskName} ]deleted successfully."}

@app.get("/user_tasks/")
async def get_user_tasks(request: Request):
    user_email = tokenChecker(request)
    user_model_dir = os.path.join(USER_MODEL_PATH, user_email)
    
    if not os.path.exists(user_model_dir):
        return {"user_tasks": []}
    
    # 搜索所有模型文件
    models = [f.split('-')[1].split('.')[0] for f in os.listdir(user_model_dir) if f.endswith(".tflite")]

    return {"user_tasks": models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
