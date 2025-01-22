'''
sift高精度特征检测的场景
orb 高效的特征提取算法，适用于实时应用，并且能够处理旋转和尺度变化。
'''


import cv2
import numpy as np

# 加载两帧图像
img1 = cv2.imread('/home/ubuntu/machine_learning/images/photo1.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('/home/ubuntu/machine_learning/images/photo2.jpg', cv2.IMREAD_COLOR)

# 初始化 SIFT/ORB/SURF
sift = cv2.ORB_create()  # OR: cv2.ORB_create() / cv2.xfeatures2d.SURF_create()

# 检测特征点并计算描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 特征点匹配
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 绘制匹配结果（可选）
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('SURF_matches.jpg', matched_img)
# 提取匹配点
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 计算单应性矩阵
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 应用透视变换拼接图像
height, width, _ = img2.shape
result = cv2.warpPerspective(img1, H, (width * 2, height))
cv2.imwrite('SURFimages1.jpg',result)
result[0:height, 0:width] = img2

cv2.imwrite('SURFstitched.jpg', result)
