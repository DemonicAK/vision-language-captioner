# training/Dockerfile
FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

COPY training/requirements.txt .
RUN pip install -r requirements.txt

COPY training/ .

CMD ["python", "train.py"]
