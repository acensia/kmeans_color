from _color_test import *
from _rembg_for_list import *


maskfolder = rembg_remove("./color_test_image", only_mask=True)

color_test("./color_test_image", maskfolder)