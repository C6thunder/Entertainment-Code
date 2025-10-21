import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def fft_draw(img_path,
             N_point=600,
             N_arrows=80,
             show_boundary=True,
             cycle=2):
    """
    使用傅里叶旋转向量绘制图形边界动画（包括内部轮廓）
    参数:
        img_path: 本地图片路径
        N_point: 动画帧数
        N_arrows: 傅里叶向量数量
        show_boundary: 是否预览边界
        cycle: 动画循环次数 (-1为无限)
    """
    # 1. 读取图片
    picture = cv2.imread(img_path)
    if picture is None:
        raise FileNotFoundError(f"图片不存在或无法读取: {img_path}")
    
    # 2. 预处理（灰度 → 二值化 → 去噪）
    img = preprocess_image(picture)
    
    # 3. 提取所有边界（包括内部轮廓）
    boundaries = find_boundaries(img, show_boundary)
    print(f"成功识别到 {len(boundaries)} 个轮廓，共 {sum(len(b) for b in boundaries)} 个边界点，开始绘制动画...")

    # 4. 绘制动画
    draw_animation(boundaries, N_point, N_arrows, cycle)


def preprocess_image(img):
    """ 图像预处理：灰度化、二值化、去噪 """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)  # 反色（黑底白图）

    # 去除小噪点
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary


def find_boundaries(img, show=False):
    """ 提取所有轮廓（包括内部）并确保是闭合曲线 """
    # 使用RETR_TREE模式获取所有轮廓，包括内部轮廓
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        raise ValueError("未检测到图形，请使用白底黑图案的图片")
    
    boundaries = []
    for contour in contours:
        # 过滤面积过小的轮廓
        if cv2.contourArea(contour) < 50:
            continue
            
        # 简化轮廓
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        
        boundary = contour.squeeze()  # Nx2
        
        # 确保边界是闭合的
        if len(boundary) > 1 and not np.allclose(boundary[0], boundary[-1]):
            boundary = np.vstack([boundary, boundary[0]])
            
        boundaries.append(boundary)
    
    if show:
        plt.figure("边界预览")
        plt.imshow(img, cmap='gray')
        for boundary in boundaries:
            plt.plot(boundary[:, 0], boundary[:, 1], 'r-', linewidth=2)
        plt.title("识别到的图形边界（包括内部，2秒后关闭）")
        plt.gca().invert_yaxis()  # 匹配Matplotlib坐标系
        plt.pause(2)
        plt.close()

    return boundaries


def draw_animation(boundaries, N_point, N_arrows, cycle):
    """ 使用傅里叶系数绘制动画，只使用一个旋转变量 """
    # 合并所有边界点用于傅里叶变换
    all_points = np.vstack(boundaries)
    
    # 转换坐标系（OpenCV到Matplotlib）
    all_points[:, 1] = -all_points[:, 1]  # y轴翻转
    
    # 中心化 & 归一化
    center = np.mean(all_points, axis=0)
    all_points = all_points - center
    scale = np.max(np.sqrt(np.sum(all_points **2, axis=1))) * 1.5
    if scale == 0:
        raise ValueError("边界点过于集中，无法生成有效动画")
    all_points = all_points / scale

    # 对每个边界进行同样的中心化和归一化
    normalized_boundaries = []
    for boundary in boundaries:
        b = boundary.copy()
        b[:, 1] = -b[:, 1]  # y轴翻转
        b = b - center
        b = b / scale
        normalized_boundaries.append(b)
    
    X, Y = all_points[:, 0], all_points[:, 1]
    n = len(X)
    
    # 傅里叶变换
    coeff_X = np.fft.fft(X)
    coeff_Y = np.fft.fft(Y)
    
    # 选择最重要的频率分量（只保留最显著的）
    magnitudes = np.abs(coeff_X) + np.abs(coeff_Y)
    sorted_indices = np.argsort(-magnitudes)
    idx = sorted_indices[:N_arrows]  # 只取前N_arrows个最显著的分量

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('black')
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')

    # 为每个轮廓创建轨迹线
    trace_lines = []
    for _ in normalized_boundaries:
        line, = ax.plot([], [], color='lime', lw=1.5)
        trace_lines.append(line)
    
    end_point, = ax.plot([], [], 'ro', markersize=5)
    vector_lines = [ax.plot([], [], color='white', lw=0.7)[0] for _ in range(len(idx))]

    # 为每个轮廓存储轨迹
    all_trace_x = [[] for _ in normalized_boundaries]
    all_trace_y = [[] for _ in normalized_boundaries]

    def update(frame):
        # 单一旋转角度（所有向量使用同一个旋转变量）
        theta = 2 * np.pi * (frame / N_point)
        
        current_x, current_y = 0, 0
        # 更新所有向量
        for i, k in enumerate(idx):
            # 计算频率（只使用一个旋转变量theta）
            freq = k if k <= n//2 else k - n
            
            # 傅里叶系数标准化
            cX = coeff_X[k] / n
            cY = coeff_Y[k] / n
            
            # 计算向量分量（使用同一个theta旋转）
            vec_x = np.real(cX * np.exp(1j * freq * theta))
            vec_y = np.real(cY * np.exp(1j * freq * theta))
            
            # 更新向量终点
            end_x = current_x + vec_x
            end_y = current_y + vec_y
            vector_lines[i].set_data([current_x, end_x], [current_y, end_y])
            current_x, current_y = end_x, end_y

        # 更新所有轮廓的轨迹
        for i, boundary in enumerate(normalized_boundaries):
            # 找到当前帧对应的边界点
            t = frame / N_point
            point_idx = int(t * (len(boundary) - 1)) % len(boundary)
            x, y = boundary[point_idx]
            
            all_trace_x[i].append(x)
            all_trace_y[i].append(y)
            
            # 限制轨迹长度
            if len(all_trace_x[i]) > N_point:
                all_trace_x[i].pop(0)
                all_trace_y[i].pop(0)
                
            trace_lines[i].set_data(all_trace_x[i], all_trace_y[i])

        end_point.set_data([current_x], [current_y])

        return vector_lines + trace_lines + [end_point]

    # 计算动画循环次数
    ani = FuncAnimation(fig, update, frames=N_point, interval=15,
                        blit=True, repeat=cycle == -1, repeat_delay=1000)
    plt.show()


# 示例运行
if __name__ == "__main__":
    fft_draw("/you/img_path",  # 替换为你的图片路径
             N_point=800,
             N_arrows=120,  # 适当增加向量数量以提高精度
             show_boundary=True,
             cycle=2)
