from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
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


app = FastAPI()

MODEL_FILE_PATH = "exported_model_test/"  # 訓練好的模型檔案路徑
UPLOAD_FOLDER = "uploads"
# 訓練狀態idle:空閒, training:訓練中, completed:訓練完成, failed:訓練失敗
TRAINING_STATUS = {"status": "idle", "accuracy": None}  

@app.get("/")
async def main():
    return FileResponse('index.html')

@app.on_event("startup")
async def startup_event():
    # 清空資料夾
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


async def save_uploaded_file(file: UploadFile, category_name: str) -> str:
    # 建立圖片分類資料夾
    category_path = os.path.join(UPLOAD_FOLDER, category_name)
    os.makedirs(category_path, exist_ok=True)
    file_location = os.path.join(category_path, file.filename)

    # 異步寫入檔案
    async with aiofiles.open(file_location, "wb") as buffer:
        await buffer.write(await file.read())

    return file_location

@app.post("/uploadfiles/")
#上傳圖片，一定要有檔案和類別名稱
async def upload_files(files: List[UploadFile] = File(...), category_name: str = Body(...)):
    if category_name is None:
        raise HTTPException(status_code=422, detail="Category name cannot be empty")
    
    print(len(files)) # 這裡會印出上傳的檔案數量
    for file in files:
        await save_uploaded_file(file, category_name)
    
    return {"message": f"Files uploaded successfully to category: {category_name}"}

@app.get("/labels/")
async def get_labels():
    """get image category labels"""
    return {"labels": os.listdir(UPLOAD_FOLDER)}

@app.post("/train/")
async def train_model(background_tasks: BackgroundTasks, training_params: TrainingParams=Body(...)):
    global TRAINING_STATUS
    if TRAINING_STATUS["status"] == "training":
        return {"message": "Model is already training."}
    elif TRAINING_STATUS["status"] == "completed":
        return {"message": "Model has already been trained."}
    
    TRAINING_STATUS["status"] = "training"
    background_tasks.add_task(train_model_task,training_params)
    return {"message": "Model training started in the background"}


async def train_model_task(training_params: TrainingParams):
    global TRAINING_STATUS
    
    try:
        image_path = UPLOAD_FOLDER
        data = image_classifier.Dataset.from_folder(image_path)
        train_data, remaining_data = data.split(0.8)
        test_data, validation_data = remaining_data.split(0.5)
        spec = image_classifier.SupportedModels.MOBILENET_V2
        hparams = image_classifier.HParams(learning_rate=training_params.learning_rate,
                                           batch_size=training_params.batch_size,
                                           epochs=training_params.epochs,
                                           export_dir=MODEL_FILE_PATH)  # model export directory
        options = image_classifier.ImageClassifierOptions(supported_model=spec, hparams=hparams)
        model = await asyncio.to_thread(image_classifier.ImageClassifier.create,
                                        train_data=train_data,
                                        validation_data=validation_data,
                                        options=options)
        loss, accuracy = model.evaluate(test_data)

        TRAINING_STATUS["accuracy"] = accuracy
        TRAINING_STATUS["status"] = "completed"

        model.export_model()

    except Exception as e:
        TRAINING_STATUS["status"] = "failed"
        TRAINING_STATUS["accuracy"] = None
        print(f"Training failed: {e}")

@app.get("/training_status/")
async def get_training_status():
    return TRAINING_STATUS

#download model
@app.get("/download_model/")
async def download_model():
    model_path = os.path.join(MODEL_FILE_PATH,"model.tflite")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Trained model file not found.")
    
    return FileResponse(
        model_path, 
        media_type="application/octet-stream",
        filename="model.tflite"
    )
    
    



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
