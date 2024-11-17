import cv2
import numpy as np
from process_image import process_image


def niu_in_range1(image, lower_bound, upper_bound):
    """
    最简单的双循环遍历像素法，用于检查图像中的每个像素是否在指定的颜色范围内。

    Args:
        image (numpy.ndarray): 输入图像，通常为 HSV 或 BGR 格式的三维 NumPy 数组。
        lower_bound (numpy.ndarray or list or tuple): 颜色范围的下界，用于定义每个通道的最小值。
        upper_bound (numpy.ndarray or list or tuple): 颜色范围的上界，用于定义每个通道的最大值。

    Returns:
        numpy.ndarray: 返回一个与输入图像大小相同的二值掩模图像，其中符合条件的像素设置为 255（白色），其余为 0（黑色）。
    """
    # 初始化与输入图像相同大小的掩模，初始值为全0（黑色）
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 遍历图像中的每个像素点
    for y in range(height):
        for x in range(width):
            # 获取当前像素点的值
            pixel = image[y, x]

            # 检查当前像素是否在下界和上界之间
            if np.all(pixel >= lower_bound) and np.all(pixel <= upper_bound):
                # 如果在范围内，设置为 255（白色）
                mask[y, x] = 255

    return mask


def niu_in_range2(image, lower_bound, upper_bound):
    """
    在双循环遍历像素法的基础上使用向量化操作加速（基于 NumPy 的向量化操作）。

    Args:
        image (numpy.ndarray): 输入图像，通常为 HSV 或 BGR 格式的三维 NumPy 数组。
        lower_bound (numpy.ndarray or list or tuple): 颜色范围的下界，用于定义每个通道的最小值。
        upper_bound (numpy.ndarray or list or tuple): 颜色范围的上界，用于定义每个通道的最大值。

    Returns:
        numpy.ndarray: 返回一个与输入图像大小相同的二值掩模图像，其中符合条件的像素设置为 255（白色），其余为 0（黑色）。
    """
    mask = np.all((image >= lower_bound) & (image <= upper_bound), axis=-1)  # 获取符合条件的布尔值矩阵
    mask = (mask * 255).astype(np.uint8)  # 将布尔值矩阵转换为 0 和 255 的掩模
    return mask






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

    # mask = hsv.copy()  # 拷贝一份原始图像，作为掩模来操作
    # height, width, _ = mask.shape  # 获取图像的高度和宽度
    # # 遍历图像中的每个像素点
    # for y in range(height):
    #     for x in range(width):
    #         # 获取当前像素点的值
    #         pixel = mask[y, x]
    #
    #         # 检查当前像素是否在下界和上界之间
    #         if np.all(pixel >= lower_red_1) and np.all(pixel <= upper_red_1):
    #             # 如果在范围内，设置为 255（白色）
    #             mask[y, x] = [255, 255, 255]
    #         else:
    #             # 如果不在范围内，设置为 0（黑色）
    #             mask[y, x] = [0, 0, 0]









    mask = niu_in_range2(hsv, lower_red_1, upper_red_1)









    # 显示处理后的图像
    cv2.imshow('Processed Image', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

























