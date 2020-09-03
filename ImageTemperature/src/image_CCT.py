import numpy as np

#from colormath.color_conversions import 

def sRGB_to_linearRGB(rgb):
    """ Конвертирует массив sRGB в linearRGB. Math source: http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html """

    return np.where(rgb[:] <= 0.04045,
                    rgb/12.92,
                    ((rgb + 0.055) / 1.055) ** 2.4)

def linearRGB_to_sRGB(rgb):
    """ Конвертирует массив linearRGB в sRGB. """

    # TODO: test
    return np.where(rgb[:] <= 0.003108,
                    rgb * 12.92,
                    1.055 * np.power(rgb, 1.0/2.4) - 0.55)

def linearRGB_to_xyz(r,g,b):
    """ Конвертирует цвет linearRGB в XYZ спектр. Math source: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html """

    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    RGB = np.array([r, g, b])

    return np.dot(M, RGB)

def xyz_arr_to_xy(xyz):

    sum = np.sum(xyz, axis=-1)
    sum[sum == 0] = 1   # Убирает деление на 0 в дальнейшем

    # TODO debug
    xy_x = xyz[:, 0] / sum
    xy_y = xyz[:, 1] / sum

    return np.array([xy_x, xy_y])

def __xy_to_CCT_Mccamy1992(xy_arr):

    n = (xy_arr[0] - 0.3320) / (xy_arr[1] - 0.1858)
    return -449 * n ** 3 + 3525 * n ** 2 - 6823.3 * n + 5520.33

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

def get_mean_CCT(rgb_img_array, mask=None, alg="Mccamy1992"):
    """ Возвращает среднее значение коллерированной цветовой температуры изображения.
    
    rgb_img_array - numpy массив пикселей формата sRGB в диапазоне [0; 1];
    mask - маска, по которой производится расчет. Автоматически нормализуется в диапазон [0, 1].
    """

    x, y, _ = rgb_img_array.shape

    if mask == None:
        mask = np.ones([x, y])
    else:
        mask = __normalize_mask(mask)

    mean_rgb = np.mean(rgb_img_array, axis=(0,1))
    mean_rgb = np.reshape(mean_rgb, (1,1,3))

    CCT_arr = get_CCT_arr(mean_rgb, alg=alg)

    # Применение маски:
    CCT_arr = CCT_arr * mask
    #bias_arr = bias_arr * mask

    # Вычисление среднего арифметического для коллерированной цветовой температуры изображения и смещения с учетом маски:
    sum_CCTs = np.sum(CCT_arr)
    #sum_biases = np.sum(bias_arr)
    sum_multipilers = np.sum(mask)

    mean_CCT = sum_CCTs / sum_multipilers
    #mean_bias = sum_biases / sum_multipilers

    return mean_CCT

def get_CCT_arr(rgb_img_array, alg="Mccamy1992"):
    """ Возвращает значения коллерированной цветовой температуры каждого пикселя изображения. TODO: добавить возврат смещения
    
    rgb_img_array - numpy массив пикселей формата sRGB в диапазоне [0; 1];
    mask - numpy массив маски. Значение маски работает как множитель для каждого пикселя при расчете его значения CCT;
    alg - "Mccamy1992", "Hernandez1999"
    """

    # Векторизация вычисления для ускорения.
    #vectfunc = np.vectorize(get_color_CCT, signature="(m,n,3)->(m,n,2)")

    x, y, _ = rgb_img_array.shape

    # Векторищация массива для numpy.map():
    rgb_vector = rgb_img_array.reshape([x*y, 3])

    linearRGB = sRGB_to_linearRGB(rgb_vector)

    r = linearRGB[:, 0]
    g = linearRGB[:, 1]
    b = linearRGB[:, 2]
    
    xyz_arr = np.array(list(map(linearRGB_to_xyz, r, g, b)))   # Конвертация в XYZ
    xy_arr = xyz_arr_to_xy(xyz_arr)       # Конвертация в XY

    if alg == "Mccamy1992":
        CCT_arr = __xy_to_CCT_Mccamy1992(xy_arr)
    if alg == "Hernandez1999":
        CCT_arr = __xy_to_CCT_Hernandez1999(xy_arr)


    # Вычисление смещения от кривой планка на плоскости XY:
    #x0, y0 = __CCT_to_xy_Kang2002()
    #bias_arr = (((xy_arr[0,:] - x0) ** 2 + (xy_arr[1,:] - y0) ** 2) ** (1 / 2))

    # Фильтрация по диапазону CCT [min; max]:
    #indexes = np.where(np.logical_and(CCT_arr >= min, CCT_arr[:] <= max))
    #CCT_arr = CCT_arr[indexes]
    #bias_arr = bias_arr[indexes]
    #mask = mask[indexes]

    CCT_arr =np.reshape(CCT_arr, [x,y])

    return CCT_arr
    """
    # Вычисление среднего арифметического для коллерированной цветовой температуры изображения и смещения с учетом маски:
    sum_CCTs = np.sum(CCT_arr)
    #sum_biases = np.sum(bias_arr)
    sum_multipilers = np.sum(mask)

    mean_CCT = sum_CCTs / sum_multipilers
    #mean_bias = sum_biases / sum_multipilers

    #return np.array([mean_CCT, mean_bias])
    return mean_CCT"""

def __normalize_mask(mask):
    """ Нормализует маску в диапазон [0; 1]
    
    mask - numpy массив маски.
    """

    min_mask_value = np.min(mask)
    max_mask_value = np.max(mask)

    return (mask - min_mask_value)/(max_mask_value - min_mask_value)
