from PIL import Image
import numpy as np
from src import image_CCT


while True:
    filename = input("Введите название изображения -> ")
    with Image.open(filename) as image:
        img_arr = np.array(image)
        
        prob = np.where(img_arr==[0,1,3], 1, 0)

        img_arr = img_arr / 255


        mean_CCT, mean_bias = image_CCT.get_image_CCT(img_arr)

        print(mean_CCT)
        print(mean_bias)
