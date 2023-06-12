from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Tensorflow
import tensorflow as tf
import numpy as np

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers,
)


@app.get("/")
async def root():
    return {"message": "Welcome!"}


@app.post("/cek_model")
async def get_net_image_prediction():
    model = tf.saved_model.load('./model/crop_recommdation')

    input_data = {
        "N": np.array([107], dtype=np.float32),
        "P": np.array([35], dtype=np.float32),
        "K": np.array([33], dtype=np.float32),
        "temperature": np.array([27], dtype=np.float32),
        "humidity": np.array([66], dtype=np.float32),
        "ph": np.array([7], dtype=np.float32),
        "rainfall": np.array([175], dtype=np.float32)
    }

    pred = model(input_data)

    # Daftar nama kelas
    class_names = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram',
                   'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
                   'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']

    # Mencari indeks dengan nilai prediksi tertinggi
    predicted_label = pred[0]
    predicted_index = np.argmax(predicted_label)

    # Mengambil nama kelas yang sesuai dengan indeks
    predicted_class = class_names[predicted_index]

    return {
        "model-prediction": predicted_class
    }


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host="0.0.0.0", port=8080)
