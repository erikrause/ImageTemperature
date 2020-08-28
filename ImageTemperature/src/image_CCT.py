import numpy as np
import colour
from PIL import Image

from colormath.color_objects import XYZColor, sRGBColor, xyYColor
from colormath.color_conversions import convert_color
import time

def CCT_Hernandez1999(lst_x, lst_y):
    n = (lst_x-0.3366)/(lst_y-0.1735)
    CCT = (-949.86315 + 6253.80338 * np.exp(-n / 0.92159) + 28.70599 * np.exp(-n / 0.20039) + 0.00004 * np.exp(-n / 0.07125))

def get_image_CCT_np(rgb_img_array, mask=None):

    x, y, _ = rgb_img_array.shape

    if mask == None:
        mask = np.ones([x, y])
    else:
        mask = normalize_mask(mask)


def get_color_CCT(rgb_color):
    """ Возвращает коррелированную цветовую температуру (CCT) цвета RGB (sRGB, [0; 1])
    и смещение (bias) относительно кривой Планка на цветовой плоскости. 
    
    rgb_color - numpy массив формата sRGB 
    """
    
    # Conversion to tristimulus values.
    #start1 = time.time()
    #XYZ = colour.sRGB_to_XYZ(rgb_color / 255)
    #end1 = time.time() - start1

    rgb = sRGBColor(rgb_color[0], rgb_color[1], rgb_color[2])
    #start2 = time.time()
    xyY = convert_color(rgb, xyYColor, target_illuminant='d65')
    x = xyY.xyy_x
    y = xyY.xyy_y
    #end2 = time.time() - start2

    # Conversion to chromaticity coordinates.
    #start3 = time.time()
    #x, y = colour.XYZ_to_xy(XYZ)
    #end3 = time.time() - start3

    # Conversion to correlated colour temperature in K.
    #CCT = colour.temperature.xy_to_CCT([x, y])#, method="Hernandez1999")

    #x0, y0 = colour.CCT_to_xy(CCT)#, method="Hernandez1999")
    #bias = ((x - x0) ** 2 + (y - y0) ** 2) ** (1 / 2)

    #return np.array([CCT, bias])
    return np.array([x, y])

def get_image_CCT(rgb_img_array, mask=None):
    """ Возвращает среднее значение коллерированной цветовой температуры изображения и смещения.
    
    rgb_img_array - numpy массив пикселей формата sRGB в диапазоне [0; 255];
    mask - numpy массив маски. Значение маски работает как множитель для каждого пикселя при расчете его значения CCT.
    """
    x, y, _ = rgb_img_array.shape

    if mask == None:
        mask = np.ones([x, y])
    else:
        mask = normalize_mask(mask)

    # Векторизация вычисления для ускорения.
    #vectfunc = np.vectorize(get_color_CCT, signature="(m,n,3)->(m,n,2)")
    #CCT_arr, bias_arr = vectfunc(rgb_img_array) * mask
    CCT_bias_mask = np.apply_along_axis(get_color_CCT, -1, rgb_img_array)
    CCT_arr = CCT_bias_mask[:,:,0]
    bias_arr = CCT_bias_mask[:,:,1]

    # Вычисление среднего арифметического для коллерированной цветовой температуры изображения и смещения с учетом маски.
    sum_CCTs = np.sum(CCT_arr)
    sum_biases = np.sum(bias_arr)
    sum_multipilers = np.sum(mask)

    mean_CCT = sum_CCTs / sum_multipilers
    mean_bias = sum_biases / sum_multipilers

    return np.array([mean_CCT, mean_bias])

def normalize_mask(mask):
    """ Нормализует маску в диапазон [0; 1]
    
    mask - numpy массив маски.
    """

    min_mask_value = np.min(mask)
    max_mask_value = np.max(mask)

    return (mask - min_mask_value)/(max_mask_value - min_mask_value)