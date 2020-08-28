import numpy as np
import colour
from PIL import Image

from colormath.color_objects import XYZColor, sRGBColor, xyYColor
from colormath.color_conversions import convert_color
import time

def sRGB_to_xyz(r,g,b):
    """ Конвертирует цвет sRGB в XYZ спектр. Math source: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html """

    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    RGB = np.array([r, g, b])

    return np.dot(M, RGB)

def xyz_arr_to_xy(xyz):

    sum = np.sum(xyz, axis=-1)
    sum[sum == 0] = 1   # Убирает деление на 0 в дальнейшем

    xy_x = xyz[:, 0] / sum
    xy_y = xyz[:, 1] / sum

    return np.array([xy_x, xy_y])

def CCT_Hernandez1999(xy_arr):
    """ Возвращает коррерированную цветовую температуру по алгоритму Hernandez et al. 1999 """

    n = (xy_arr[0]-0.3366)/(xy_arr[1]-0.1735)
    CCT = (-949.86315 + 
           6253.80338 * np.exp(-n / 0.92159) + 
           28.70599 * np.exp(-n / 0.20039) + 
           0.00004 * np.exp(-n / 0.07125))
    #CCT[0] = 77000

    n = np.where(CCT > 50000,
                 (xy_arr[0] - 0.3356) / (xy_arr[1] - 0.1691),
                 n)

    CCT = np.where(CCT > 50000,
                   36284.48953 + 0.00228 * np.exp(-n / 0.07861) +
                   5.4535e-36 * np.exp(-n / 0.01543),
                   CCT)

    return CCT
 
def CCT_to_xy_Kang2002(CCT_arr):
    """ Kan et al. 2002. Допустимый диапазон: [1667, 25000] """

    # TODO: проверить на всем диапазоне
    x = np.where(CCT_arr <= 4000,
                 -0.2661239 * 10 ** 9 / CCT_arr ** 3 -
                 0.2343589 * 10 ** 6 / CCT_arr ** 2 +
                 0.8776956 * 10 ** 3 / CCT_arr +
                 0.179910,
                 -3.0258469 * 10 ** 9 / CCT_arr ** 3 +
                 2.1070379 * 10 ** 6 / CCT_arr ** 2 +
                 0.2226347 * 10 ** 3 / CCT_arr +
                 0.24039)

    y = np.select([CCT_arr <= 2222,
                   np.logical_and(CCT_arr > 2222, CCT_arr <= 4000),
                   CCT_arr > 4000],
                  [-1.1063814 * x ** 3 -
                   1.34811020 * x ** 2 +
                   2.18555832 * x -
                   0.20219683,
                   -0.9549476 * x ** 3 -
                   1.37418593 * x ** 2 +
                   2.09137015 * x -
                   0.16748867,
                   3.0817580 * x ** 3 -
                   5.8733867 * x ** 2 +
                   3.75112997 * x -
                   0.37001483])

    return np.array([x, y])


def CCT_to_xyz_Hernandez1999(CCT_arr):
    """" Конвертирует цветовую температуру в спектр XY """

    return colour.CCT_to_xy(CCT_arr, method="Hernandez1999")

def get_color_CCT(rgb_color):
    """ Возвращает коррелированную цветовую температуру (CCT) цвета RGB (sRGB, [0; 1])
    и смещение (bias) относительно кривой Планка на цветовой плоскости. 
    
    rgb_color - numpy массив формата sRGB 
    """
    
    # Conversion to tristimulus values.
    start1 = time.time()
    #XYZ = colour.sRGB_to_XYZ(rgb_color / 255)
    

    rgb = sRGBColor(rgb_color[0], rgb_color[1], rgb_color[2])
    #start2 = time.time()
    xyY = convert_color(rgb, xyYColor, target_illuminant='d65')
    x = xyY.xyy_x
    y = xyY.xyy_y
    end1 = time.time() - start1
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
    #CCT_bias_mask = np.apply_along_axis(get_color_CCT, -1, rgb_img_array)
    #CCT_arr = CCT_bias_mask[:,:,0]
    #bias_arr = CCT_bias_mask[:,:,1]

    start1 = time.time()

    # Векторищация массива для numpy.map():
    rgb_vector = rgb_img_array.reshape([x*y, 3])
    r = rgb_vector[:, 0]
    g = rgb_vector[:, 1]
    b = rgb_vector[:, 2]
    mask = mask.reshape([x*y])

    xyz_arr = np.array(list(map(sRGB_to_xyz, r, g, b)))
    xy_arr = xyz_arr_to_xy(xyz_arr)
    end1 = time.time() - start1
    CCT_arr = CCT_Hernandez1999(xy_arr) * mask
    x0, y0 = CCT_to_xy_Kang2002(CCT_arr)
    bias_arr = (((xy_arr[0,:] - x0) ** 2 + (xy_arr[1,:] - y0) ** 2) ** (1 / 2)) * mask

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