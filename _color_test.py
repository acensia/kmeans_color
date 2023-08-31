import cv2
import numpy as np

import json

with open("./hex_map.json", 'r') as j:
    c_dict = json.load(j)
with open("./kmeans_map.json", 'r') as j:
    k_dict = json.load(j)

def show_colors(c, wind):    
    colors = []
    for rgb in c:
        print(rgb)
        paper = np.full((200, 200, 3), rgb, dtype=np.uint8)
        colors.append(paper)
    total = np.hstack(colors)
    print(total)
    total = cv2.cvtColor(total, cv2.COLOR_RGB2BGR)
    cv2.imshow(wind, total)
    return
    
    
def rgb_to_hex(rgb): # convert rgb array to hex code
    return "{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

# kmeans function by 최준혁
from sklearn.cluster import KMeans

def cvt_216(pxs):  # convert rgb into the value in 216
    st = [0, 51, 102, 153, 204, 255]
    p0 = min(st, key=lambda x : abs(x - pxs[0]))
    p1 = min(st, key=lambda x : abs(x - pxs[1]))
    p2 =min(st, key=lambda x : abs(x - pxs[2]))

    return np.array((p0, p1, p2))

def cvt_closest(pxs):
    dict = k_dict
    min_distance = float('inf')
    closest = -1
    for i in dict:
        point = np.array(dict[i], dtype=float)
        distance = np.linalg.norm(np.array(point) - pxs)
        if distance < min_distance:
            min_distance = distance
            closest = i
    
    return closest, dict[closest]
        
        

def pixels_argsort(li): # argsort pixels in list, must be 3 channels
    return np.unique(li, axis=0, return_counts=True)

def dom_with_Kmeans(img_list, k=3): #use kmeans
    pixels = img_list
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    # Get the colors
    colors = kmeans.cluster_centers_
    # Get the labels (which cluster each pfixel belongs to)
    labels = kmeans.labels_    
    # Count the frequency of each label
    label_counts = np.bincount(labels)
    
    # Find the most frequent label
    # return colors, labels
    # dominant_label = np.argmax(label_counts)
    # Get the dominant color
    # dominant_color = colors[dominant_label]

    labels = np.argsort(label_counts)[::-1] # argsort for cluster labels

    dom_counts = [label_counts[i] for i in labels[:3]] 
    total = sum(dom_counts)
    dom_counts = [d/total for d in dom_counts] # each cluster's rate

    dom_colors = [colors[d_lab] for d_lab in labels[:3]] # 3 most colors
    
    return dom_colors, dom_counts

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

def classify_color(image_path, mask_img=None, imshow_check=False):
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
    else: # [0,0,0] 이외의 pixel값만 뽑아서 list (거의 폐기)
        cropped_list = np.array([image_rgb[i][j] for i in range(height) for j in range(width) if not (np.array_equal(image_rgb[i][j], np.array([0,0,0])) or np.array_equal(image_rgb[i][j], np.array([255, 255, 255])))])
        

    # list 내 에서 가장 많은 색상 value 추출 관련
    unique_colors, color_counts = pixels_argsort(cropped_list)
    
        
    # # 가장 많은 색상 및 두 번째로 많은 색상 추출
    # sorted_indices = np.argsort(color_counts)[::-1] # [3, 1, 0, 2, 4][::-1] -> [4, 2, 0, 1, 3]
    # dominant_color = unique_colors[sorted_indices[0]]
    # second_dominant_color = unique_colors[sorted_indices[1]]
    # # 완전한 흰색과 검은색인 경우에 대한 처리
    # if np.array_equal(dominant_color, [255, 255, 255]) or np.array_equal(dominant_color, [0, 0, 0]):
    #     dominant_color = second_dominant_color
    # 가장 유사한 색상 찾기
    # min_distance = float('inf')
    # classified_color = None
    # dis = []  # color별 거리 저장
    # c_d = []  # color -> label_number 매핑
    # for color_name, color_value in colors.items():
    #     distance = np.linalg.norm(np.array(color_value) - dominant_color)
    #     dis.append(distance)
    #     c_d.append(color_name)
    #     if distance < min_distance:
    #         min_distance = distance
    #         classified_color = color_name

    # dist = np.argsort(np.array(dis))[:-1]
    
    # 전체 pixels rgb 평균값  by 최준혁
    # mean_color_R = np.mean(cropped_list[:, 0])
    # mean_color_G = np.mean(cropped_list[:, 1])
    # mean_color_B = np.mean(cropped_list[:, 2])
    # mean_color = np.array([mean_color_R, mean_color_G, mean_color_B])
    # print(mean_color_R, mean_color_G, mean_color_B)

    kmeans_color, kmeans_dis = dom_with_Kmeans(cropped_list) # Kmeans by 최준혁
    # print(kmeans_color)

    dominant_color = kmeans_color[0]
    fst, snd, trd = kmeans_color[:3]
    print(kmeans_dis)
    print(fst)
    print(snd)
    print(trd)
    
    # # get average color with clusters' rate  
    # aver_color = [fst[0]*kmeans_dis[0] + snd[0]*kmeans_dis[1] + trd[0]*kmeans_dis[2], fst[1]*kmeans_dis[0] + snd[1]*kmeans_dis[1] + trd[1]*kmeans_dis[2], fst[2]*kmeans_dis[0] + snd[2]*kmeans_dis[1] + trd[2]*kmeans_dis[2]]

    fst_cvt = cvt_216(fst)
    snd_cvt = cvt_216(snd)
    trd_cvt = cvt_216(trd)
    
    
    # Show the image
    if imshow_check:
        cv2.imshow("cloth",cv2.cvtColor(image_rgb,cv2.COLOR_RGB2BGR))
        show_colors([fst, snd, trd], "og")
        show_colors([fst_cvt, snd_cvt, trd_cvt], "cvt")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Show the imag
    # return classified_color
    # return f, s, t
    return c_dict[rgb_to_hex(fst_cvt)], c_dict[rgb_to_hex(snd_cvt)], c_dict[rgb_to_hex(trd_cvt)]


## 모든 test 파일 color check해서 test_res 디렉토리에 저장 #by 최준혁
import os
import glob


def color_test(foldername, maskfolder, image_type="*.png", imshow_check=False):

    test_img = [f for f in glob.glob(os.path.join(foldername,image_type))]

    os.makedirs(f"{foldername}_kmeans", exist_ok=True)
    # os.makedirs("./colors", exist_ok=True)

    for i,t in enumerate(test_img):
        test_num = os.path.basename(t)[:-4]
        img = cv2.imread(t)

        mask_img = f"{maskfolder}\\{test_num}_mask.png"

        pred_color, p2, p3 = classify_color(t, mask_img=mask_img, imshow_check=imshow_check)
        print(i, pred_color, p2, p3)
        # cv2.imshow(pred_color, img)
        cv2.imwrite(f"{foldername}_kmeans\\{pred_color}_{test_num}.png", img)





