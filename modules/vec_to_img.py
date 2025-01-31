from haio import img_to_url
from PIL import Image
from tensorflow.keras.datasets import cifar10
import io
import numpy as np
import pyperclip


def vec_to_img(img_vec: np.ndarray) -> bytes:
    img_vec = img_vec.astype(np.uint8, copy=False)
    img = Image.fromarray(img_vec)
    output = io.BytesIO()
    img.save(output, format="PNG", compress_level=9, pnginfo=None)
    return output.getvalue()


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    img = vec_to_img(x_train[0])
    url = img_to_url(img_data=img, mime_type="image/png")

    print(url[:50] + "...")
    pyperclip.copy(url)
