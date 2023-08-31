import os, glob
import cv2


cloth_list = glob.glob(os.path.join("./color_test_sets", "*.png")) + glob.glob(os.path.join("./color_test_sets", "*.jpg"))

os.makedirs("./resized_testset", exist_ok=True)
for f in cloth_list:
    # print(f)
    img = cv2.imread(f)
    img = cv2.resize(img, dsize=(400, 400), interpolation=cv2.INTER_AREA)
    filename = os.path.basename(f)[:-4]
    # print(filename)
    cv2.imwrite(f"./resized_testset/{filename}_resizsed.jpg", img)