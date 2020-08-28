from PIL import Image
import numpy as np
from src import image_CCT

method = "pixel"

while True:
    filename = input("Введите название изображения -> ")
    with Image.open(filename) as image:
        img_arr = np.array(image)

        if method == "img":
            mean_CCT, mean_bias = image_CCT.get_image_CCT(img_arr)
        if method == "pixel":
            mean_color = np.mean(img_arr, axis=(0,1))
            mean_CCT, mean_bias = image_CCT.get_color_CCT(mean_color)

        print(mean_CCT)
        print(mean_bias)
