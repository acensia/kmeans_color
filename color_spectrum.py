import numpy as np
import cv2

# Dimensions of the paper (height, width)

avail = [0, 51, 102, 153, 204, 255]
img = np.full((300, 1, 3), [0,0,0], dtype=np.uint8)
for i in avail:
    for j in avail:
        for k in avail:
            img = np.hstack((img, np.full((300, 3, 3), [i, j, k], dtype=np.uint8)))

print(img.shape)

# Show the image
cv2.imshow("3-Color Paper", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("3-Color Paper",cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
