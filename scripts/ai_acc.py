import sys
import os

abspath = os.path.dirname(os.path.abspath(sys.argv[0]))
top_path = os.path.dirname(abspath)
sys.path.append(os.path.join(top_path, "modules"))

from haio import img_to_url, haio_hash
from vec_to_img import vec_to_img
import json
import tensorflow.keras.datasets.cifar10 as cifar10
import numpy as np
from PIL import Image
import io

labels = [
    "Airplane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]


def new_vec_to_img(img_vec: np.ndarray) -> bytes:
    img_vec = img_vec.astype(np.uint8, copy=False)
    img = Image.fromarray(img_vec)
    output = io.BytesIO()
    img.save(output, format="PNG", compress_level=9, pnginfo=None)
    return output.getvalue()


if __name__ == "__main__":

    data_num = 10000

    with open(
        os.path.join(top_path, "notebooks/haio_cache/1c57d50d996eb63a8be99364e27bc016"),
        "r",
        encoding="utf-8",
    ) as file:
        data_src = json.load(file)  # JSONをPythonのdictに変換

    data = {"question_template": data_src["question_template"], "data_lists": {}}

    (x_train, y_train), _ = cifar10.load_data()

    acc_counts = {"llama": 0, "openai": 0, "nova": 0, "gemini": 0}

    for i in range(data_num):
        img = x_train[i]
        img_url = img_to_url(vec_to_img(img), mime_type="image/png")
        answer_list = data_src["data_lists"][haio_hash([img_url])]["answer_list"]

        for client in acc_counts.keys():
            first_ans_info: dict = next(
                (v for v in answer_list.values() if v.get("client") == client), {}
            )
            if first_ans_info.get("answer") == labels[y_train[i][0]]:
                acc_counts[client] += 1

    for client, acc in acc_counts.items():
        print(f"{client}: {acc / data_num}")
