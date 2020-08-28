import unittest
import numpy as np
from src import image_CCT
import math

class Test_image_CCT(unittest.TestCase):

    def test_white_color_CCT(self):
        """ Тест температуры и смещения для белого цвета с погрешностью 5% """

        CCT, bias = image_CCT.get_color_CCT(np.array([1, 1, 1]))

        self.assertTrue((CCT > 6200) and (CCT < 6800), msg="Цветовая температура белого пикселя за пределами погрешности")
        self.assertTrue((bias < math.fabs(0.05)), msg="Смещение XY белого пикселя за пределами погрешности")

    def test_normalize_mask(self):
        """ Тест нормализации маски. """

        mask = np.array([-25, 42])
        normalized_mask = image_CCT.normalize_mask(mask)

        self.assertTrue((normalized_mask[0] == 0) and (normalized_mask[1] == 1))

    def test_white_image_CCT(self):
        """ Тест температуры и смещения для белого изображения с погрещность 5%"""

        img_arr = np.ones(shape=[100,100,3])
        
        CCT, bias = image_CCT.get_image_CCT(img_arr)

        self.assertTrue((CCT > 6200) and (CCT < 6800), msg="Цветовая температура белого изображения за пределами погрешности")
        self.assertTrue((bias < math.fabs(0.05)), msg="Смещение XY белого изображения за пределами погрешности")


if __name__ == '__main__':
    unittest.main()
