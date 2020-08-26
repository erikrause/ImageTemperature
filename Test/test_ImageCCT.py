import unittest
import numpy as np
from src import Image_CCT
import math

class Test_image_CCT(unittest.TestCase):

    def test_white_pixel_CTT(self):
        """ Тест температуры и смещения для белого цвета с погрешностью 5% """

        CCT, bias = Image_CCT.get_pixel_CCT(np.array([255, 255, 255]))

        self.assertTrue((CCT > 6200) and (CCT < 6800), msg="Цветовая температура белого цвета за пределами погрешности")
        self.assertTrue((bias < math.fabs(0.05)), msg="Смещение XY белого цвета за пределами погрешности")

    def test_normalize_mask(self):
        """ Тест нормализации маски. """

        mask = np.array([-25, 42])
        normalized_mask = Image_CCT.normalize_mask(mask)

        self.assertTrue((normalized_mask[0] == 0) and (normalized_mask[1] == 1))

if __name__ == '__main__':
    unittest.main()
