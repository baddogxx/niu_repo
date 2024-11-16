import cv2
import numpy as np
from process_image import process_image


def main():
    # 读取图像
    image_path = r'E:\git_repositories\niu_repo\red_ball.jpg'
    frame = cv2.imread(image_path)

    # 检查是否成功读取图像
    if frame is None:
        print("Error: Could not load image.")
        return

    # 处理图像，识别红色物体并叠加方框
    #processed_frame = process_image(frame)

    # setp1
    # BGR到HSV转换
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # setp2
    # 定义红色的 HSV 范围
    lower_red_1 = np.array([0, 120, 70])  # 第一个红色下界值：色调0、饱和度120、亮度70
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 120, 70])  # 第二个红色界限
    upper_red_2 = np.array([180, 255, 255])

    # setp3
    # 根据 HSV 范围构建mask，检测红色
    # mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)    #将图像中低于下界和高于上界的像素变为0，其余变为255
    # mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    # mask = mask1 | mask2

    mask = hsv.copy()  # 拷贝一份原始图像，作为掩模来操作
    height, width, _ = mask.shape  # 获取图像的高度和宽度
    # 遍历图像中的每个像素点
    for y in range(height):
        for x in range(width):
            # 获取当前像素点的值
            pixel = mask[y, x]

            # 检查当前像素是否在下界和上界之间
            if np.all(pixel >= lower_red_1) and np.all(pixel <= upper_red_1):
                # 如果在范围内，设置为 255（白色）
                mask[y, x] = [255, 255, 255]
            else:
                # 如果不在范围内，设置为 0（黑色）
                mask[y, x] = [0, 0, 0]











    # 显示处理后的图像
    cv2.imshow('Processed Image', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

























