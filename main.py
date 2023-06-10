from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import tensorflow_decision_forests
import tensorflow as tf

from tensorflow import expand_dims
from tensorflow.nn import softmax
import numpy as np
from numpy import argmax
from numpy import max
from numpy import array
from json import dumps

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
    return {"message": "Welcome !"}


@app.get("/cek-model")
async def get_net_image_prediction():
    model = tf.saved_model.load('./model/crop_recommdation')

    input_data = {
        "N": np.array([107]),
        "P": np.array([35]),
        "K": np.array([33]),
        "temperature": np.array([27]),
        "humidity": np.array([66]),
        "ph": np.array([7]),
        "rainfall": np.array([175])
    }

    pred = model.serve(input_data)

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
