import sys
import os

abspath = os.path.dirname(os.path.abspath(sys.argv[0]))
top_path = os.path.dirname(abspath)
sys.path.append(os.path.join(top_path, "modules"))

from haio import img_to_url, haio_hash
from vec_to_img import vec_to_img
import hashlib
import json
import tensorflow.keras.datasets.cifar10 as cifar10
import numpy as np
from PIL import Image
import io


def new_vec_to_img(img_vec: np.ndarray) -> bytes:
    img_vec = img_vec.astype(np.uint8, copy=False)
    img = Image.fromarray(img_vec)
    output = io.BytesIO()
    img.save(output, format="PNG", compress_level=9, pnginfo=None)
    return output.getvalue()


if __name__ == "__main__":
    with open(
        os.path.join(top_path, "notebooks/haio_cache/1c57d50d996eb63a8be99364e27bc016"),
        "r",
        encoding="utf-8",
    ) as file:
        data_src = json.load(file)  # JSONをPythonのdictに変換

    data = {"question_template": data_src["question_template"], "data_lists": {}}

    (x_train, y_train), _ = cifar10.load_data()

    for i in range(10000):
        img = x_train[i]
        old_img_url = img_to_url(vec_to_img(img), mime_type="image/png")
        answer_list = data_src["data_lists"][haio_hash([old_img_url])]["answer_list"]
        new_img_url = img_to_url(new_vec_to_img(img), mime_type="image/png")
        data["data_lists"][haio_hash([new_img_url])] = {
            "data_list": [new_img_url],
            "answer_list": answer_list,
        }
        # print(hashlib.md5(new_vec_to_img(x_train[i])).hexdigest())

    with open(
        os.path.join(top_path, "notebooks/haio_cache/fixed"), "w", encoding="utf-8"
    ) as file:
        json.dump(data, file, ensure_ascii=False)
