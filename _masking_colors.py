from _color_test import *
from _rembg_for_list import *


maskfolder = rembg_remove("./resized_testset", only_mask=True, image_type="*.jpg")

color_test("./resized_testset", maskfolder, imshow_check=True, image_type="*.jpg")