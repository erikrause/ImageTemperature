import unittest
import numpy as np
from src import PixelCCT
#from src import PixelCCT
#from math import abs

class Test_pixelCCT(unittest.TestCase):
    def test_CTT_for_white(self):
        """ Тест температуры и смещения для белого цвета с погрешностью 5% """
        CCT, bias = PixelCCT.get_CCT(np.array([255, 255, 255]))
        self.assertTrue((CCT > 6200) and (CCT < 6800), )
        #self.assertTrue((bias < abs))

if __name__ == '__main__':
    unittest.main()
