import unittest
import numpy as np
from src import image_CCT
from src.image_CCT import __linearRGB_to_xyz as _image_CCT__linearRGB_to_xyz
from src.image_CCT import __xyz_arr_to_xy as _image_CCT__xyz_arr_to_xy
from src.image_CCT import __xy_to_CCT_Hernandez1999 as _image_CCT__xy_to_CCT_Hernandez1999
from src.image_CCT import __sRGB_to_linear_RGB as _image_CCT__sRGB_to_linear_RGB

import math
from PIL import Image

import colour
import colormath
import colormath.color_conversions
from colormath.color_objects import XYZColor, sRGBColor, xyYColor


class Test_image_CCT(unittest.TestCase):

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

    def test_CCT_by_mean_pixel(self):

        img = Image.open("l6400.png")

    def test_xy(self):

        rgb = np.array([0.0, 0.0039215686274509803, 0.011764705882352941])
        r, g, b = _image_CCT__sRGB_to_linear_RGB(rgb)

        xyz = _image_CCT__linearRGB_to_xyz(r,g,b)
        rgb_obj = sRGBColor(rgb[0], rgb[1], rgb[2])
        xyz2 = colormath.color_conversions.RGB_to_XYZ(rgb_obj)
        xyz3 = colour.sRGB_to_XYZ(rgb)

        #x = 0.18724231852549086	
        #y = 0.1940723233616907

        xy = _image_CCT__xyz_arr_to_xy(np.expand_dims(xyz, 0))
        #xy2 = _image_CCT__xyz_arr_to_xy(np.swapaxes(np.array([[xyz2.xyz_x], [xyz2.xyz_y], [xyz2.xyz_z]]), 0, -1))
        xy2 = colormath.color_conversions.XYZ_to_xyY(xyz2)
        xy3 = colour.XYZ_to_xy(xyz3)

        CCT = _image_CCT__xy_to_CCT_Hernandez1999(xy)
        #CCT2 = _image_CCT__xy_to_CCT_Hernandez1999(xy2)

        x = xy[0,0]
        y = xy[1,0]
        real = colour.temperature.xy_to_CCT_Hernandez1999(np.array([x, y]))
        real3 = colour.temperature.xy_to_CCT_Hernandez1999(xy3)
        real4 = colour.temperature.xy_to_CCT_McCamy1992(xy3)
        real5 = colour.temperature.xy_to_CCT_Kang2002(xy3)

        obr = colour.temperature.CCT_to_xy_Hernandez1999(xy3)

        prob = 0

if __name__ == '__main__':
    unittest.main()
