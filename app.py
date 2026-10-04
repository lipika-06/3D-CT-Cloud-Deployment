from flask import Flask, render_template, request
import numpy as np
import os

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("ct_file")

    if file is None or file.filename == "":
        return render_template(
            "index.html",
            result="Please select a file."
        )

    # Create a small synthetic 3D CT volume
    size = 32

    image = np.random.normal(
        0, 0.1, (size, size, size)
    )

    # Create a lung-like region
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

    # Simple abnormal region
    image[12:20, 12:18, 12:18] += 1

    # Simple segmentation
    segmented = image > 0.3

    # Calculate abnormal percentage
    abnormal_pixels = np.sum(segmented)
    total_pixels = segmented.size

    abnormal_percentage = (
        abnormal_pixels / total_pixels
    ) * 100

    # Simple classification
    if abnormal_percentage > 5:
        result = "ABNORMAL CT"
    else:
        result = "NORMAL CT"

    return render_template(
        "index.html",
        result=result,
        probability=round(
            abnormal_percentage, 2
        )
    )


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 10000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
