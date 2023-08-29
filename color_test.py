import cv2
import numpy as np


# kmeans function by 최준혁
from sklearn.cluster import KMeans

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
	return hist

def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	
	# return the bar chart
	return bar

def dom_with_Kmeans(img_list, k=3):
    pixels = img_list
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the colors
    colors = kmeans.cluster_centers_
    
    # Get the labels (which cluster each pixel belongs to)
    labels = kmeans.labels_
    
    # Count the frequency of each label
    label_counts = np.bincount(labels)
    
    # Find the most frequent label
    # return colors, labels
    # dominant_label = np.argmax(label_counts)
    # Get the dominant color
    # dominant_color = colors[dominant_label]

    labels = np.argsort(label_counts)[::-1]
	
    dom_colors = [colors[d_lab] for d_lab in labels[:3]]
    
    return dom_colors

# Example usage


# 미리 정의한 색상과 해당 색상에 대한 레이블
colors = {
    'white' : (255, 255, 255),
    'black' : (32, 32, 32),
    'gray' : (145, 145, 145),
    'beige' : (225, 198, 153),    
    'red' : (200, 45, 45),
    'yellow' : (240, 240, 50),
    'khaki' : (148, 129, 43),
    'navy' : (2, 7, 93),
    'green' : (9, 148, 65),
    'brown' : (153, 76, 20),
    'skyblue' : (2, 204, 254),
    'pink' : (255, 182, 193),
    'purple' : (106, 40, 173)
}

def classify_color(image_path, mask_img=None):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환

    # 이미지 중앙 부분 크기 및 마진 설정
    center_margin = 0.4  # 이미지 가로 및 세로 크기의 % 만큼을 중앙 부분으로 사용
    height, width, _ = image_rgb.shape
    center_size = int(min(height, width) * (1 - center_margin * 2))

    # 중앙 부분 자르기
    # center_x = width // 2
    # center_y = height // 2
    # cropped_image = image_rgb[center_y - center_size // 2:center_y + center_size // 2,
    #                           center_x - center_size // 2:center_x + center_size // 2]

    # crop with background mask by 최준혁
    if mask_img: # mask 흰색 영역 내의 pixel값만 뽑아서 list
        mask = cv2.imread(mask_img)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cropped_list = np.array([image_rgb[i][j] for i in range(height) for j in range(width) if mask[i][j] > 100])
    else: # [0,0,0] 이외의 pixel값만 뽑아서 list
        cropped_list = np.array([image_rgb[i][j] for i in range(height) for j in range(width) if not (np.array_equal(image_rgb[i][j], np.array([0,0,0])) or np.array_equal(image_rgb[i][j], np.array([255, 255, 255])))])
        

    # 이미지에서 가장 많은 색상 추출
    reshaped_image = cropped_list.reshape(-1, 3) 
    unique_colors, color_counts = np.unique(reshaped_image, axis=0, return_counts=True)
    # set = [1, 2, 3, 4, 5],  cnts = [2, 2, 3, 1, 6]

    # rgb 평균값  by 최준혁
    # mean_color_R = np.mean(cropped_list[:, 0])
    # mean_color_G = np.mean(cropped_list[:, 1])
    # mean_color_B = np.mean(cropped_list[:, 2])
    # mean_color = np.array([mean_color_R, mean_color_G, mean_color_B])
    # print(mean_color_R, mean_color_G, mean_color_B)
    
    # 가장 많은 색상 및 두 번째로 많은 색상 추출
    sorted_indices = np.argsort(color_counts)[::-1] # [3, 1, 0, 2, 4][::-1] -> [4, 2, 0, 1, 3]
    dominant_color = unique_colors[sorted_indices[0]]
    second_dominant_color = unique_colors[sorted_indices[1]]

    print(dominant_color) # dominant 확인 by 최준혁


    # 완전한 흰색과 검은색인 경우에 대한 처리
    if np.array_equal(dominant_color, [255, 255, 255]) or np.array_equal(dominant_color, [0, 0, 0]):
        dominant_color = second_dominant_color

    # dominant_color = dom_with_Kmeans(cropped_list) # Kmeans by 최준혁
    kmeans_color = dom_with_Kmeans(cropped_list)
    print(kmeans_color)

    dominant_color = kmeans_color[0]
    cv2.imshow("cloth",cv2.cvtColor(image_rgb,cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Dimensions of the paper (height, width)
    height = 300
    width = 250

    # Blue color in BGR format (since OpenCV uses BGR instead of RGB)
    # Create a blank image with blue color
    # k1 = np.full((height, width, 3), kmeans_color[0], dtype=np.uint8)
    # k2 = np.full((height, width, 3), kmeans_color[1], dtype=np.uint8)
    # k3 = np.full((height, width, 3), kmeans_color[2], dtype=np.uint8)
    # kk = np.hstack((k1, k2, k3))

    # # Show the image
    # cv2.imshow("kk", cv2.cvtColor(kk,cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # 가장 유사한 색상 찾기
    # min_distance = float('inf')
    # classified_color = None
    # dis = []
    # c_d = []
    # for color_name, color_value in colors.items():
    #     distance = np.linalg.norm(np.array(color_value) - dominant_color)
    #     dis.append(distance)
    #     c_d.append(color_name)
    #     if distance < min_distance:
    #         min_distance = distance
    #         classified_color = color_name

    # dist = np.argsort(np.array(dis))[:-1]

    # return classified_color, dist, c_d
    
    st = [0, 51, 102, 153, 204, 255]
    fst, snd, trd = kmeans_color[:3]
    def nearest(px):
        return min(st, key=lambda x : abs(x - px))
    fst_nearest = np.array([nearest(fst[0]), nearest(fst[1]), nearest(fst[2])])
    snd_nearest = np.array([nearest(snd[0]), nearest(snd[1]), nearest(snd[2])])
    trd_nearest = np.array([nearest(trd[0]), nearest(trd[1]), nearest(trd[2])])
    k1 = np.full((height, width, 3), fst_nearest, dtype=np.uint8)
    k2 = np.full((height, width, 3), snd_nearest, dtype=np.uint8)
    k3 = np.full((height, width, 3), trd_nearest, dtype=np.uint8)
    kk = np.hstack((k1, k2, k3))

    # Show the image
    cv2.imshow("kk", cv2.cvtColor(kk,cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    return fst_nearest
    

# image_path = "test13.png"  # 분석할 이미지 경로
# predicted_color = classify_color(image_path)
# print(f"분류된 색상: {predicted_color}")


## 모든 test 파일 color check해서 test_res 디렉토리에 저장 #by 최준혁
import os
import glob
test_img = [f for f in glob.glob(os.path.join("./color_test_image","*.png"))]
# print(test_img.size)
# os.rmdir("./test_res_rembg")
os.makedirs("./test_res_rembg", exist_ok=True)
for t in test_img:
    test_num = os.path.basename(t)[:-4]
    img = cv2.imread(t)

    mask_img = f"./rembg_mask\\{test_num}_mask.png"

    pred_color = classify_color(t, mask_img=mask_img)
    print(pred_color)
    # cv2.imshow(pred_color, img)
    # cv2.imwrite(f"./test_res_rembg\\{pred_color}_{test_num}.png", img)





