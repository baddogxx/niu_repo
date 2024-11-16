# import cv2
#
# print(cv2.getVersionString())


# import cv2
# # 读取图片
# image = cv2.imread(r'C:\Users\baddog\Pictures\Saved Pictures\niu.jpg')
# # 显示图片
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np


def main():
    # 读取图像
    image_path = r'E:\git_repositories\niu_repo\red_ball.jpg'  # 请修改为实际图片路径
    frame = cv2.imread(image_path)

    # 检查是否成功读取图像
    if frame is None:
        print("Error: Could not load image.")
        return

    # 处理图像，识别红色物体并叠加方框
    processed_frame = process_image(frame)

    # 显示处理后的图像
    cv2.imshow('Processed Image', processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像处理函数，识别红色物体并叠加方框
def process_image(frame):
    # 将图像从 BGR 转换为 HSV 色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义红色的 HSV 范围
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])

    # 根据 HSV 范围构建掩模，检测红色
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = mask1 | mask2

    # 使用掩模提取红色区域
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 找到红色物体的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原图上叠加方框
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 过滤掉较小的噪声轮廓
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色方框

    return frame


if __name__ == "__main__":
    main()
