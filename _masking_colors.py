from _modules._color_test import *
from _modules._rembg_for_list import *


img_path = f"./color_test_image"

maskfolder = rembg_remove_list(img_path, only_mask=True, image_type="*.*")

color_test_list(img_path, maskfolder, imshow_check=True, image_type="*.*")


# import cv2
# img = cv2.imread("./sample_input.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# mask = rembg_remove(img, True)
# pred = color_test(img, mask=mask)
# print(pred)