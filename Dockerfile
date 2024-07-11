# 使用 python:3.12-slim 映像作為基礎映像
FROM python:3.10-slim as base

# 創建一個名為 builder 的新階段，並使用 base 映像作為基礎映像
FROM base as builder

# 設定工作目錄為 /app
WORKDIR /app

# 複製 requirements.txt 到工作目錄
COPY requirements.txt ./

# 複製當前目錄下的所有文件到容器中的 /app 目錄
COPY . ./

# 安裝 Python 套件，並將 pip 的快取目錄掛載為 Docker 的快取
RUN pip install --no-cache-dir -r requirements.txt
RUN yes | pip uninstall opencv-python
RUN pip install --no-cache-dir opencv-python-headless

# 創建 exported_model_test 目錄並設置適當的權限
RUN mkdir -p /app/exported_model_test && chown -R root:root /app/exported_model_test
RUN mkdir -p /app/uploads && chown -R root:root /app/uploads

# 設定 exported_model_test 目錄為VOLUME，以便在容器重新啟動時保存模型
VOLUME /app/exported_model_test
VOLUME /app/uploads

EXPOSE 8000
# 當 Docker 容器啟動時，執行 Uvicorn 伺服器
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["gunicorn", "main:app"]
