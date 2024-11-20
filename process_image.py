import cv2
import numpy as np

def process_image(frame):


# setp1
    # BGR到HSV转换
    hsv = np.zeros_like(frame, dtype=np.float32)
    frame = frame.astype(np.float32) / 255.0  # 归一化
    B, G, R = frame[..., 0], frame[..., 1], frame[..., 2]
    Cmax = np.maximum(np.maximum(R, G), B)
    Cmin = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin
    H = np.zeros_like(Cmax)
    mask = (delta != 0)
    H[(mask) & (Cmax == R)] = ((G - B) / delta)[(mask) & (Cmax == R)] % 6
    H[(mask) & (Cmax == G)] = ((B - R) / delta)[(mask) & (Cmax == G)] + 2
    H[(mask) & (Cmax == B)] = ((R - G) / delta)[(mask) & (Cmax == B)] + 4
    H *= 60
    H[H < 0] += 360
    S = np.zeros_like(Cmax)
    S[Cmax != 0] = (delta / Cmax)[Cmax != 0]
    V = Cmax
    hsv[..., 0] = H / 2  # OpenCV 中 H 的范围是 [0, 179]
    hsv[..., 1] = S * 255
    hsv[..., 2] = V * 255

    # 转换为 uint8 类型，以便 OpenCV 函数使用
    hsv = hsv.astype(np.uint8)

# setp2
    # 定义红色的 HSV 范围
    lower_red_1 = np.array([0, 120, 70])        #第一个红色下界值：色调0、饱和度120、亮度70
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 120, 70])      #第二个红色界限
    upper_red_2 = np.array([180, 255, 255])

# setp3
    # 根据 HSV 范围构建mask，检测红色
    # mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)    #将图像中低于下界和高于上界的像素变为0，其余变为255
    # mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    # mask = mask1 | mask2

    mask = frame
    height, width, _ = mask.shape      #获取图像的高度和宽度
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





    # 使用掩模提取红色区域
    result = cv2.bitwise_and(frame, frame, mask=mask)       #图像按位与

    # 找到红色物体的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原图上叠加方框
    frame = (frame * 255).astype(np.uint8)  # 将 frame 转换回 uint8 格式
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 过滤掉较小的噪声轮廓
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色方框

    return frame





#
# import cv2
# import numpy as np
#
# def process_image(frame): 
#     # 将图像从 BGR 转换为 HSV 色彩空间
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # 定义红色的 HSV 范围
#     lower_red_1 = np.array([0, 120, 70])
#     upper_red_1 = np.array([10, 255, 255])
#     lower_red_2 = np.array([170, 120, 70])
#     upper_red_2 = np.array([180, 255, 255])
#
#     # 根据 HSV 范围构建掩模，检测红色
#     mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
#     mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
#     mask = mask1 | mask2
#
#     # 使用掩模提取红色区域
#     result = cv2.bitwise_and(frame, frame, mask=mask)
#
#     # 找到红色物体的轮廓
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 在原图上叠加方框
#     for contour in contours:
#         if cv2.contourArea(contour) > 500:  # 过滤掉较小的噪声轮廓
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色方框
#
#     return frame
#
