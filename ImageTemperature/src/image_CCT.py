import numpy as np
import colour
from PIL import Image

def get_pixel_CCT(rgb_pixel):
    """ Возвращает коррелированную цветовую температуру (CCT) пикселя RGB (sRGB, [0; 255])
   и смещение (bias) относительно кривой Планка на цветовой плоскости. 
   
   rgb_pixel - numpy массив формата sRGB 
   """

    # Conversion to tristimulus values.
    XYZ = colour.sRGB_to_XYZ(rgb_pixel / 255)

    # Conversion to chromaticity coordinates.
    x, y = colour.XYZ_to_xy(XYZ)
    #x, y = [0.1, 0.4]

    # Conversion to correlated colour temperature in K.
    CCT = colour.temperature.xy_to_CCT([x, y], method="Hernandez1999")

    x0, y0 = colour.CCT_to_xy(CCT, method="Hernandez1999")
    bias = ((x - x0) ** 2 + (y - y0) ** 2) ** (1 / 2)

    return CCT, bias

def get_image_CCT(rgb_img_array, mask=None):
    """ Возвращает среднее значение коллерированной цветовой температуры изображения по маске 
    
    rgb_img_array - numpy массив пикселей формата sRGB в диапазоне [0; 255];
    mask - numpy массив маски. Значение маски работает как множитель для каждого пикселя при расчете среднего значения CCT.
    """
    x, y, _ = rgb_img_array.shape

    if mask == None:
        mask = np.ones([x, y])
    else:
        mask = normalize_mask(mask)

    # Вычисление среднего арифметического: image_CCT = pixel_CCT_sum / pixel_multipiler_sum
    pixel_CCT_sum = 0
    pixel_multipiler_sum = 0
    for i in range(0, x):
        for j in range(0, y):
            pixel = rgb_img_array[i][j]
            pixel_multipiler = mask[i][j]

            pixel_CCT = get_pixel_CCT(pixel) * pixel_multipiler
            pixel_CCT_sum += pixel_CCT
            pixel_multipiler_sum += pixel_multipiler

    image_CCT = pixel_CCT_sum / pixel_multipiler_sum

    return image_CCT


def normalize_mask(mask:type(np.ndarray)):
    """ Нормализует маску в диапазон [0; 1]
    
    mask - numpy массив маски.
    """

    min_mask_value = np.min(mask)
    max_mask_value = np.max(mask)

    return (mask - min_mask_value)/(max_mask_value - min_mask_value)