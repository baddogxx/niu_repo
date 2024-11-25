import cv2
import numpy as np
import concurrent.futures
import time
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


def niu_in_range3(image, lower_bound, upper_bound, block_size=(256, 256)):
    """
    使用并行化和分块处理的方式，结合向量化操作，检查图像中的每个像素是否在指定的颜色范围内。

    Args:
        image (numpy.ndarray): 输入图像，通常为 HSV 或 BGR 格式的三维 NumPy 数组。
        lower_bound (numpy.ndarray or list or tuple): 颜色范围的下界，用于定义每个通道的最小值。
        upper_bound (numpy.ndarray or list or tuple): 颜色范围的上界，用于定义每个通道的最大值。
        block_size (tuple): 分块的大小，默认为 (32, 32)。

    Returns:
        numpy.ndarray: 返回一个与输入图像大小相同的二值掩模图像，其中符合条件的像素设置为 255（白色），其余为 0（黑色）。
    """
    height, width = image.shape[:2]
    block_height, block_width = block_size
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    def process_block(y, x):
        """处理每个块的函数"""
        block = image[y:y + block_height, x:x + block_width]
        block_mask = np.all((block >= lower_bound) & (block <= upper_bound), axis=-1)
        return (y, x, block_mask)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 遍历图像的每个小块并并行处理
        futures = []
        for y in range(0, height, block_height):
            for x in range(0, width, block_width):
                futures.append(executor.submit(process_block, y, x))

        # 更新掩模
        for future in concurrent.futures.as_completed(futures):
            y, x, block_mask = future.result()
            mask[y:y + block_height, x:x + block_width] = np.where(block_mask, 255,
                                                                   mask[y:y + block_height, x:x + block_width])

    return mask

def niu_in_range4(image, lower_bound, upper_bound):
    """
    使用 OpenCV CUDA 加速检查图像中的每个像素是否在指定的颜色范围内。

    Args:
        image (numpy.ndarray): 输入图像，通常为 HSV 或 BGR 格式的三维 NumPy 数组。
        lower_bound (numpy.ndarray or list or tuple): 颜色范围的下界，用于定义每个通道的最小值。
        upper_bound (numpy.ndarray or list or tuple): 颜色范围的上界，用于定义每个通道的最大值。

    Returns:
        numpy.ndarray: 返回一个与输入图像大小相同的二值掩模图像，其中符合条件的像素设置为 255（白色），其余为 0（黑色）。
    """
    # 将图像从 CPU 传输到 GPU
    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(image)

    # 将颜色范围也从 CPU 传输到 GPU
    gpu_lower = cv2.cuda_GpuMat()
    gpu_lower.upload(lower_bound)
    gpu_upper = cv2.cuda_GpuMat()
    gpu_upper.upload(upper_bound)

    # 使用 GPU 执行颜色范围检查
    gpu_mask = cv2.cuda.inRange(gpu_image, gpu_lower, gpu_upper)

    # 将结果从 GPU 下载回 CPU
    mask = gpu_mask.download()

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
    start_time = time.time()                                #开始计时
    mask1 = niu_in_range1(hsv, lower_red_1, upper_red_1)
    end_time = time.time()                                  #结束计时
    print(f"The time it takes is: {end_time - start_time:.4f} seconds")

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

    #print(cv2.getBuildInformation())
    # 显示处理后的图像
    cv2.imshow('Processed Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

























