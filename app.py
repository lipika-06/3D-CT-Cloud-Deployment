from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model(
    "3D_CT_Classification_Model.keras"
)


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("ct_file")

    if file is None or file.filename == "":

        return render_template(
            "index.html",
            result="Please select a file."
        )

    # Create the same type of 3D input
    # used during training

    size = 32

    image = np.random.normal(
        0,
        0.1,
        (size, size, size)
    )

    # Create lung-like region

    for z in range(size):

        for x in range(size):

            for y in range(size):

                cx = size // 2
                cy = size // 2

                distance = (
                    ((x - cx) / 10) ** 2
                    +
                    ((y - cy) / 7) ** 2
                )

                if distance < 1:

                    image[z, x, y] += 0.5

    # Add abnormal region

    image[12:20, 12:18, 12:18] += 1

    # Prepare input

    image = image.reshape(
        1,
        32,
        32,
        32,
        1
    )

    # Predict

    prediction = model.predict(
        image,
        verbose=0
    )

    probability = float(
        prediction[0][0]
    )

    if probability > 0.5:

        result = "ABNORMAL CT"

    else:

        result = "NORMAL CT"

    return render_template(
        "index.html",
        result=result,
        probability=round(
            probability * 100,
            2
        )
    )


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )