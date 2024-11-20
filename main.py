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


def niu_bitwise_and(image1, image2, mask=None):
    """
    使用自定义函数实现 cv2.bitwise_and 的基本功能。

    Args:
        image1 (numpy.ndarray): 输入的第一幅图像。
        image2 (numpy.ndarray): 输入的第二幅图像，通常与第一幅图像相同。
        mask (numpy.ndarray, optional): 掩模图像，决定哪些像素进行按位与操作。

    Returns:
        numpy.ndarray: 按位与操作后的结果图像。
    """
    # 确保输入图像大小相同
    assert image1.shape == image2.shape, "输入图像大小必须相同"

    # 初始化输出图像，与输入图像大小相同
    result = np.zeros_like(image1, dtype=np.uint8)

    # 获取图像的高度和宽度
    height, width = image1.shape[:2]

    # 遍历图像中的每个像素点
    for y in range(height):
        for x in range(width):
            # 如果没有掩模或者掩模中对应位置的值为 255，则进行按位与操作
            if mask is None or mask[y, x] == 255:
                result[y, x] = image1[y, x] & image2[y, x]

    return result


def niu_find_mask_center(mask):
    """
    根据二值掩模图像计算包含物体的中心点坐标。

    Args:
        mask (numpy.ndarray): 二值掩模图像，像素值为 0 或 255。

    Returns:
        tuple: 包含物体的中心点的 (x, y) 坐标。
    """
    # 获取图像的高度和宽度
    height, width = mask.shape[:2]

    # 初始化一些变量来计算中心点
    total_x = 0
    total_y = 0
    total_points = 0

    # 遍历图像中的每个像素点
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 255:
                total_x += x
                total_y += y
                total_points += 1

    # 如果找到了白色像素点，计算中心点坐标
    if total_points > 0:
        cX = total_x // total_points
        cY = total_y // total_points
        return (cX, cY)
    else:
        return None




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
    # 定义红色的 HSV 范围
    mask1 = niu_in_range2(hsv, lower_red_1, upper_red_1)
    mask2 = niu_in_range2(hsv, lower_red_2, upper_red_2)
    mask = mask1 | mask2

    #step4
    # 根据掩模计算中心点
    center = niu_find_mask_center(mask)
    if center is not None:
        cX, cY = center
        # 打印中心点坐标
        print(f"Center of the object: ({cX}, {cY})")
        # 在原始图像上绘制中心点
        cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
        cv2.putText(frame, "Center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



    # 显示处理后的图像
    cv2.imshow('Processed Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

























