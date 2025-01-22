'''
用于行人检测
'''
import cv2
import numpy as np

# 读取图像
image = cv2.imread('/home/ubuntu/machine_learning/images/photo4.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 创建 HOG 特征描述符
hog = cv2.HOGDescriptor()

# 加载 OpenCV 中预训练的行人检测器
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 检测行人
# detectMultiScale 用于检测图像中的所有行人，返回检测到的行人的位置
boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

# 应用非最大抑制（NMS），去除重叠的矩形框
indices = cv2.groupRectangles(boxes, 1, 0.2)[0]

# 绘制去重后的矩形框
for (x, y, w, h) in indices:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Pedestrian Detection with NMS', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

