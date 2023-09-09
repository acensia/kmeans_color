import os, glob
import cv2


re_size = (200, 200)

cloth_list = glob.glob(os.path.join("./color_test_sets", "*.png")) + glob.glob(os.path.join("./color_test_sets", "*.jpg"))

os.makedirs(f"./resized_testset_{re_size}", exist_ok=True)
for f in cloth_list:
    # print(f)
    img = cv2.imread(f)
    img = cv2.resize(img, dsize=re_size, interpolation=cv2.INTER_AREA)
    filename = os.path.basename(f)[:-4]
    # print(filename)
    cv2.imwrite(f"./resized_testset_{re_size}/{filename}_resizsed_{re_size}.jpg", img)