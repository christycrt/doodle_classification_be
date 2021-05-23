import uvicorn
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

app = FastAPI()

origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# main model
doodle_model = tf.keras.models.load_model('doodle_model_v1.h5')
doodle_classes = ['apple', 'axe', 'baseball', 'bicycle', 'car', 'carrot', 'chair', 'compass', 'donut', 'house', 'ice cream', 'pants', 'pig', 'sailboat', 'scissors', 't-shirt']

@app.post("/predict/")
async def predict_doodle_class(file: bytes = File(...), doodle_class1: str = "", doodle_class2: str = ""):
    img = Image.open(io.BytesIO(file))
    img = img.convert("L")
    img = img.resize((28, 28))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # plt.imshow(img_array, cmap=plt.cm.binary)
    # plt.show()
    data_np_array = np.array([img_array])

    result, results = predictDoodleClass(data_np_array)
    check_result = False
    if result['predict_class'] == doodle_class1:
        check_result = True
    elif result['predict_class'] == doodle_class2:
        check_result = True
    return {"result":check_result,"result_predict": result, "results": results}

@app.get("/doodle-classes/")
async def get_doodle_classes():
    return {"doodle_classes":doodle_classes}

def predictDoodleClass(image):
    predictions = doodle_model.predict(image)
    predict_class = np.argmax(predictions[0])
    predict_class_percent = predictions[0][predict_class]
    results = []
    i = 0
    for prediction in predictions[0]:
        results.append({"accuracy":str(prediction), "predict_class":doodle_classes[i]})
        i += 1
    return {"accuracy": str(predict_class_percent), "predict_class":doodle_classes[predict_class]}, results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)