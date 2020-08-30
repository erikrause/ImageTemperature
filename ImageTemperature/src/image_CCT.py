import numpy as np
import colour
from PIL import Image

#from colormath.color_conversions import 

def __sRGB_to_linear_RGB(rgb):
    """ Конвертирует массив sRGB в linearRGB. Math source: http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html """

    return np.where(rgb[:] <= 0.04045,
                    rgb/12.92,
                    ((rgb + 0.055) / 1.055) ** 2.4)

def __linearRGB_to_xyz(r,g,b):
    """ Конвертирует цвет linearRGB в XYZ спектр. Math source: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html """

    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    RGB = np.array([r, g, b])

    return np.dot(M, RGB)

def __xyz_arr_to_xy(xyz):

    sum = np.sum(xyz, axis=-1)
    sum[sum == 0] = 1   # Убирает деление на 0 в дальнейшем

    # TODO debug
    xy_x = xyz[:, 0] / sum
    xy_y = xyz[:, 1] / sum

    return np.array([xy_x, xy_y])

def __xy_to_CCT_Hernandez1999(xy_arr):
    """ Возвращает коррерированную цветовую температуру по алгоритму Hernandez et al. 1999 """

    n = (xy_arr[0]-0.3366)/(xy_arr[1]-0.1735)
    CCT = (-949.86315 + 
           6253.80338 * np.exp(-n / 0.92159) + 
           28.70599 * np.exp(-n / 0.20039) + 
           0.00004 * np.exp(-n / 0.07125))
    #CCT[0] = 77000

    prob = CCT > 50000
    n1 = np.where(CCT > 50000,
                 (xy_arr[0] - 0.3356) / (xy_arr[1] - 0.1691),
                 n)

    CCT2 = np.where(CCT > 50000,
                   36284.48953 + 0.00228 * np.exp(-n1 / 0.07861) +
                   5.4535e-36 * np.exp(-n1 / 0.01543),
                   CCT)

    return CCT
 
def __CCT_to_xy_Kang2002(CCT_arr):
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

def get_image_CCT(rgb_img_array, mask=None):
    """ Возвращает среднее значение коллерированной цветовой температуры изображения и смещения.
    
    rgb_img_array - numpy массив пикселей формата sRGB в диапазоне [0; 1];
    mask - numpy массив маски. Значение маски работает как множитель для каждого пикселя при расчете его значения CCT.
    """
    x, y, _ = rgb_img_array.shape

    if mask == None:
        mask = np.ones([x, y])
    else:
        mask = __normalize_mask(mask)

    # Векторизация вычисления для ускорения.
    #vectfunc = np.vectorize(get_color_CCT, signature="(m,n,3)->(m,n,2)")

    # Векторищация массива для numpy.map():
    rgb_vector = rgb_img_array.reshape([x*y, 3])
    mask = mask.reshape([x*y])

    linearRGB = __sRGB_to_linear_RGB(rgb_vector)

    r = linearRGB[:, 0]
    g = linearRGB[:, 1]
    b = linearRGB[:, 2]
    
    xyz_arr = np.array(list(map(__linearRGB_to_xyz, r, g, b)))   # Конвертация в XYZ
    xy_arr = __xyz_arr_to_xy(xyz_arr)       # Конвертация в XY
    CCT_arr = __xy_to_CCT_Hernandez1999(xy_arr) * mask      # Получение температуры для каждого пикселя

    # Вычисление смещения от кривой планка на плоскости XY
    x0, y0 = __CCT_to_xy_Kang2002(CCT_arr)      
    bias_arr = (((xy_arr[0,:] - x0) ** 2 + (xy_arr[1,:] - y0) ** 2) ** (1 / 2)) * mask

    # Вычисление среднего арифметического для коллерированной цветовой температуры изображения и смещения с учетом маски.
    sum_CCTs = np.sum(CCT_arr)
    sum_biases = np.sum(bias_arr)
    sum_multipilers = np.sum(mask)

    mean_CCT = sum_CCTs / sum_multipilers
    mean_bias = sum_biases / sum_multipilers

    return np.array([mean_CCT, mean_bias])

def __normalize_mask(mask):
    """ Нормализует маску в диапазон [0; 1]
    
    mask - numpy массив маски.
    """

    min_mask_value = np.min(mask)
    max_mask_value = np.max(mask)

    return (mask - min_mask_value)/(max_mask_value - min_mask_value)