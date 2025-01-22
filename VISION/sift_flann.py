import cv2
import numpy as np

# 加载两张待配准的图像
img1 = cv2.imread('/home/ubuntu/machine_learning/images/photo1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/home/ubuntu/machine_learning/images/photo2.jpg', cv2.IMREAD_GRAYSCALE)

# 使用 SIFT 提取特征点
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 定义 FLANN 参数
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # 检查树的次数，值越大越精确但越慢

# 初始化 FLANN 匹配器
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 进行特征匹配
matches = flann.knnMatch(des1, des2, k=2)  # 每个描述符找 2 个最近邻

# 应用比值测试（Lowe's Ratio Test）过滤错误匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # 比值测试阈值，可调节
        good_matches.append(m)

# 可视化匹配结果
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('/home/ubuntu/machine_learning/FLANN_matches.jpg', img_matches)

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 计算单应性矩阵
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 应用透视变换拼接图像
height, width = img2.shape
result = cv2.warpPerspective(img1, H, (width * 2, height))
cv2.imwrite('FLANNimages1.jpg',result)
result[0:height, 0:width] = img2

cv2.imwrite('FLANNstitched.jpg', result)