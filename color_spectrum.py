import numpy as np
import cv2

# Dimensions of the paper (height, width)

avail = [0, 51, 102, 153, 204, 255]
img = np.array([[[int(i/42)*42, int(j/51)*42, int(k/51)*42] for k in range(256) for j in range(256)] for i in range(256)], dtype=np.uint8)

print(img.shape)

# Show the image
cv2.imshow("3-Color Paper", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("3-Color Paper",cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
