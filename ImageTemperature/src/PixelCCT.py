import numpy as np
import colour

def get_CCT(pixel):
    """Возвращает коррелированную цветовую температуру пикселя RGB (sRGB, диапазон 0-255)"""

    # Conversion to tristimulus values.
    XYZ = colour.sRGB_to_XYZ(pixel / 255)

    # Conversion to chromaticity coordinates.
    x, y = colour.XYZ_to_xy(XYZ)
    #x, y = [0.1, 0.4]

    # Conversion to correlated colour temperature in K.
    CCT = colour.temperature.xy_to_CCT([x, y], method="Hernandez1999")

    x0, y0 = colour.CCT_to_xy(CCT, method="Hernandez1999")
    bias = ((x - x0) ** 2 + (y - y0) ** 2) ** (1 / 2)

    return CCT, bias
