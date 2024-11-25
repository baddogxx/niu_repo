- [基于python的物体标记](#基于python的物体标记)
  - [1.项目简介](#1项目简介)
  - [2.算法流程](#2算法流程)
    - [2.1 色彩空间转换](#21-色彩空间转换)
      - [2.1.1 定义红色HSV范围](#211-定义红色hsv范围)
    - [2.2 构建掩膜](#22-构建掩膜)
    - [2.3 查找轮廓](#23-查找轮廓)
    - [2.4 叠加标记](#24-叠加标记)
  - [3.加速](#3加速)
    - [3.1 算法优化](#31-算法优化)
      - [3.1.1 向量化操作](#311-向量化操作)
      - [3.1.2 分块处理与并行化](#312-分块处理与并行化)
      - [3.1.3 动态规划](#313-动态规划)
  - [4.小小感悟](#4小小感悟)
# 基于python的物体标记
## 1.项目简介
本项目基于python实现了简易的物体标记算法。通过对比同种算法实现的不同方式，来探寻数字图像处理中的加速问题。  
物体标记算法演示demo中使用了一张红色小球主体，白色背景的测试图像（如下图所示），在测试图片中小球作为ROI区域与背景的主要区别是其颜色，人眼可以通过这一点将其与背景区分开来。本项目通过算法处理能够在图片中以绿色方框标记红色小球轮廓并返回小球中心点坐标。下面介绍具体算法流程：   
![alt text](red_ball.jpg)
## 2.算法流程
最基本的处理算法使用OpenCV库函数实现，打包封装后的处理函数如下：
```
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
```
- [基于python的物体标记](#基于python的物体标记)
  - [1.项目简介](#1项目简介)
  - [2.算法流程](#2算法流程)
    - [2.1 色彩空间转换](#21-色彩空间转换)
      - [2.1.1 定义红色HSV范围](#211-定义红色hsv范围)
    - [2.2 构建掩膜](#22-构建掩膜)
    - [2.3 查找轮廓](#23-查找轮廓)
    - [2.4 叠加标记](#24-叠加标记)
  - [3.加速](#3加速)
    - [3.1 算法优化](#31-算法优化)
      - [3.1.1 向量化操作](#311-向量化操作)
      - [3.1.2 分块处理与并行化](#312-分块处理与并行化)
      - [3.1.3 动态规划](#313-动态规划)
  - [4.小小感悟](#4小小感悟)
### 2.1 色彩空间转换
首先使用cv2.imread()函数读入本地路径中的jpg格式的图像。
    示例代码中使用OpenCV库中的cv2.cvtColor()函数实现了图像的RGB色彩空间到HSV色彩空间的转化，传入参数选择预设类型cv2.COLOR_BGR2HSV将BGR图像转换为HSV图像。这样可以更方便地进行颜色阈值的操作，尤其是在进行颜色提取时，HSV色彩空间相比BGR更加直观和适用。
  
#### 2.1.1 定义红色HSV范围
通过定义两个红色的 HSV 范围来检测图像中的红色物体。由于红色在 HSV 空间中分布在 0 和 180 之间，因此需要分别定义低范围和高范围。
lower_red_1 和 upper_red_1 定义了红色的低端范围（0 到 10），而 lower_red_2 和 upper_red_2 则定义了红色的高端范围（170 到 180）。这两个范围可以确保图像中从低到高的红色区域都能被准确提取。
代码中除了使用OpenCV库中的cv2.cvtColor()函数实现了BGR色彩空间到HSV色彩的转换外，还使用了简单的双循环遍历算法的方式实现这一功能。
~~~
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
~~~
上述代码展示了双循环遍历每个像素的方式实现类似cv2.cvtColor()函数的功能。首先创建与输入图像相同大小的零矩阵，用于储存转换后的HSV图像，第二步归一化输入的RGB图像，第三步分离三色通道。接下来依次计算色相H、饱和度S、明度V。最后归一化为HSV并存储。
### 2.2 构建掩膜
  使用 cv2.inRange() 函数，根据上述定义的红色 HSV 范围来生成两个掩膜 mask1 和 mask2，分别对应低端和高端的红色范围。函数的作用是将图像中符合指定HSV 范围的区域标记为 255（白色），其他区域标记为 0（黑色）。接着，通过使用按位或运算 | 将两个掩膜合并，得到最终的红色掩膜 mask，这个掩膜能够有效区分出图像中的红色区域。
### 2.3 查找轮廓
使用 cv2.findContours() 函数在掩膜图像 mask 中查找轮廓。
该函数会返回图像中所有轮廓的坐标信息，并通过轮廓的面积来过滤掉噪声小物体。轮廓的提取为后续的目标定位提供了基础。
### 2.4 叠加标记
对于每个找到的轮廓，通过 cv2.boundingRect() 获取外接矩形的坐标及尺寸，并在原图上使用 cv2.rectangle() 函数绘制绿色的矩形框，标记出检测到的红色物体区域。通过设置一个面积阈值（比如 500），可以过滤掉一些过小的噪声区域。最终，在原始图像上显示出所有符合条件的红色物体，并返回处理后的图像。

## 3.加速

### 3.1 算法优化
#### 3.1.1 向量化操作
向量化操作在图像处理中的作用是通过将逐像素的计算转化为批量处理，从而显著提高图像处理算法的效率和性能。传统的图像处理方法通常需要对每个像素进行循环操作，而向量化通过一次性对多个像素进行批量处理，减少了计算时间并充分利用了现代处理器的并行计算能力，尤其是支持SIMD（单指令多数据）或GPU加速的硬件。常见的图像处理操作如色彩空间转换、滤波、二值化、边缘检测等，都会从向量化中获益。通过使用像OpenCV和NumPy等库中的内置函数，图像处理可以在无需手动实现复杂的向量化操作的情况下，高效地完成。向量化还可以减少内存消耗，特别是在处理大图像时，显著提升处理速度，成为实时图像处理和大规模数据分析中不可或缺的重要技术手段。  
~~~
def niu_in_range1(image, lower_bound, upper_bound):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            if np.all(pixel >= lower_bound) and np.all(pixel <= upper_bound):
                mask[y, x] = 255 
    return mask
~~~
关于二值化掩膜的构建首先使用上述函数完成，上述代码调用了numpy库实现了基本的双循环遍历算法，函数首先初始化了一个与掩膜大小相同的空白图像，接着使用image.shape函数获取输入图像的行列像素数量，然后按照行列循环遍历图像的每一个像素，循环中按照像素是否在指定的颜色范围内生成掩膜输出。
~~~
    start_time = time.time()                                #开始计时
    mask1 = niu_in_range1(hsv, lower_red_1, upper_red_1)
    end_time = time.time()                                  #结束计时
    print(f"The time it takes is: {end_time - start_time:.4f} seconds")
~~~
利用time模块实现函数运行时间的计时，该函数单次运行所需时间为4.0402秒。可以看出此方式效率很低，无法满足实时的图像处理需求。Python本身是解释型语言，这意味着每个操作都需要被逐行解释并执行，每次访问图像中的像素都会做一次内存访问。这会带来巨大的额外的开销。  
针对此方法的不足提出加速方式为向量化操作，
~~~
def niu_in_range2(image, lower_bound, upper_bound):
    mask = np.all((image >= lower_bound) & (image <= upper_bound), axis=-1)  
    mask = (mask * 255).astype(np.uint8)  
    return mask
~~~
niu_in_range2函数即利用了numpy的向量化操作实现了算法的加速，向量化操作会将整个数据集视为一个整体，利用底层的并行计算框架优化运算，通常会在硬件级别实现并行化。这意味着多个数据点可以在一个批处理内同时进行计算，从而提高速度。例如，NumPy会尽可能将所有操作加载到CPU的高速缓存中，而避免频繁的内存访问。这种内存局部性优化有效提高了处理速度，尤其是在数据量大时。
使用此方法可将函数运行时间缩短到0.0472秒，实现了效率的较大提升。
#### 3.1.2 分块处理与并行化
~~~
def niu_in_range3(image, lower_bound, upper_bound, block_size=(256, 256)):
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
            mask[y:y + block_height, x:x + block_width] = np.where(block_mask, 255, mask[y:y + block_height, x:x + block_width])

    return mask
~~~
上述函数在向量化操作的基础上加入了分块处理的思想，process_block(y, x)：是一个处理图像小块的内部函数，该函数返回包含当前块的左上角坐标和掩模数组。  
在cpu类计算平台上实现并行化处理操作常利用cpu的多线程技术，代码中使用了Python中的concurrent.futures.ThreadPoolExecutor来实现多线程操作，目的是并行地处理图像的多个小块。它为每个图像块分配了一个线程，使得图像处理可以在多个线程中同时进行，从而提高处理效率，特别是在处理大图像时能显著提高运行速度。本次测试设备使用了酷睿i7-12700cpu，8P核+4E核，共20线程，对于图像处理类的计算密集型应用，Python 的GIL（全局解释器锁）限制了多线程的并行性，即使线程数大于 CPU 核心数，Python也只能在同一时刻运行一个线程的计算。由此niu_in_range3函数的分块操作中将每个小块的大小设置为256*256，能够较均衡的适配处理器的硬件处理能力。运行此段代码总耗时为0.0130秒，相比单纯的向量化操作也有较大提升。
#### 3.1.3 动态规划
到目前为止，所有的代码都是在独立处理单帧的图像，当红色小球在背景中不断运动时，为了实现实时输出小球位置坐标的功能，可利用动态规划的思想加速整个处理过程。  

    一、图像中只有单个小球，小球所像素占整幅图像比重较小。以双循环算法实现掩膜的构建为例，无论采用何种硬件加速方式，处理器均要遍历每个像素点才能构建完整的掩膜。若能够只遍历小球及其周围的部分像素点，可大大减少遍历所需的像素点数量，提高算法的性能。关于所要遍历的像素区域的选取可利用上一帧计算出的小球中心坐标加以实现。
    二、使用先进的算法预测小球的轨迹，真实世界中没有小球会做无规则的运动，物体甚至动物的运动轨迹能够在一定时间范围内被预测，例如单个摄像头使用光流法加深度学习网络实现小球运动轨迹的精细化预测，另一个摄像头使用传统的图像处理算法计算小球位置，二者结果在时间上交替输出或卡尔曼滤波后输出。利用这些方式能够一定程度上弥补从物体运动到摄像头拍摄、设备处理输出间的延时。

## 4.小小感悟
青青子衿，悠悠我心。愿小牛：好好运动、好好学习、好好玩耍、好好吃饭、好好休息，过一个开心有趣、精神百倍的人生！

























