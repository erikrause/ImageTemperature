from PIL import Image
import numpy as np
from src import image_CCT


while True:
    filename = input("Введите название изображения -> ")
    with Image.open(filename) as image:
        img_arr = np.array(image)

        img_arr = img_arr / 255

        #mean = np.mean(img_arr, axis=(0,1))
        #mean_CCT = image_CCT.get_image_CCT(np.reshape(mean, (1,1,3)), alg="Hernandez1999")
        image_CCT_arr = image_CCT.get_CCT_arr(img_arr)
        mean_CCT = image_CCT.get_mean_CCT(img_arr)

        print(image_CCT_arr)
        print(mean_CCT)