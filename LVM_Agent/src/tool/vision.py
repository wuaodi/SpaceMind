import cv2
import numpy as np
from typing import Union, Optional
import base64
from PIL import Image
import io
import os


def _decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
    """安全地解码base64图像数据
    
    Args:
        base64_string (str): base64图像字符串
        
    Returns:
        Optional[np.ndarray]: 解码后的图像数组，失败时返回None
    """
    try:
        # 检查是否是data URL格式
        if base64_string.startswith("data:image"):
            # 提取base64数据部分
            if "," not in base64_string:
                return None
            base64_data = base64_string.split(",")[1]
        else:
            base64_data = base64_string
            
        # 清理base64字符串（移除空格和换行符）
        base64_data = base64_data.replace(" ", "").replace("\n", "").replace("\r", "")
        
        # 确保base64字符串长度是4的倍数（添加padding）
        missing_padding = len(base64_data) % 4
        if missing_padding:
            base64_data += '=' * (4 - missing_padding)
        
        # 验证base64字符串的有效性
        try:
            # 测试解码一小部分数据
            test_decode = base64.b64decode(base64_data[:20])
        except Exception:
            return None
            
        # 解码完整的base64数据
        image_data = base64.b64decode(base64_data)
        
        # 检查数据长度
        if len(image_data) < 100:  # 太小的数据不太可能是有效图像
            return None
            
        # 转换为numpy数组
        nparr = np.frombuffer(image_data, np.uint8)
        
        # 尝试解码为图像
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
        
    except Exception as e:
        print(f"base64解码错误: {str(e)}")
        return None


def extract_edges(image_path: str, low_threshold: int = 50, high_threshold: int = 150) -> str:
    """提取图像边缘，用于卫星图像分析
    
    Args:
        image_path (str): 输入图像路径或base64编码
        low_threshold (int): Canny边缘检测的低阈值，默认50
        high_threshold (int): Canny边缘检测的高阈值，默认150
    
    Returns:
        str: 边缘图像的base64编码字符串
    """
    try:
        # 输入验证
        if not image_path:
            return "错误：图像路径不能为空"
        
        # 验证阈值参数
        if not isinstance(low_threshold, int) or not isinstance(high_threshold, int):
            return "错误：阈值参数必须是整数"
        
        if low_threshold <= 0 or high_threshold <= 0:
            return "错误：阈值参数必须是正数"
        
        if low_threshold >= high_threshold:
            return "错误：低阈值必须小于高阈值"
        
        # 处理base64编码的图像
        if image_path.startswith("data:image") or (len(image_path) > 100 and not os.path.exists(image_path)):
            print("检测到base64图像数据，开始解码...")
            image = _decode_base64_image(image_path)
            
            if image is None:
                return "错误：base64图像数据无效或损坏。请检查数据完整性。"
                
            input_type = "base64"
        else:
            # 读取本地文件
            if not os.path.exists(image_path):
                return f"错误：文件不存在 {image_path}"
                
            image = cv2.imread(image_path)
            input_type = "file"
            
        if image is None:
            return f"错误：无法读取图像。可能的原因：1) 文件格式不支持 2) 文件损坏 3) base64数据无效"
        
        # 验证图像尺寸
        if image.shape[0] < 10 or image.shape[1] < 10:
            return "错误：图像尺寸太小，无法进行边缘检测"
        
        print(f"成功加载图像，尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊，减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # 将结果转换为PIL图像
        edge_image = Image.fromarray(edges)
        
        # 保存边缘图像到内存
        buffered = io.BytesIO()
        edge_image.save(buffered, format="PNG")
        
        # 转换为base64编码
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # 根据输入类型决定是否保存文件
        if input_type == "base64":
            # base64输入，不保存到文件
            return f"边缘提取成功！已从base64图像中提取边缘特征。\n边缘图像base64编码: data:image/png;base64,{img_base64[:100]}..."
        else:
            # 本地文件输入，保存边缘图像到文件
            output_path = image_path.replace('.', '_edges.')
            cv2.imwrite(output_path, edges)
            return f"边缘提取成功！边缘图像已保存至: {output_path}\n边缘图像base64编码: data:image/png;base64,{img_base64[:100]}..."
        
    except Exception as e:
        return f"边缘提取失败: {str(e)}"


def analyze_satellite_image(image_path: str) -> str:
    """分析卫星图像，提取关键特征
    
    Args:
        image_path (str): 卫星图像路径或base64编码
    
    Returns:
        str: 分析结果描述
    """
    try:
        # 处理base64编码的图像
        if image_path.startswith("data:image") or (len(image_path) > 100 and not os.path.exists(image_path)):
            image = _decode_base64_image(image_path)
            
            if image is None:
                return "错误：base64图像数据无效或损坏。请检查数据完整性。"
        else:
            # 读取本地文件
            if not os.path.exists(image_path):
                return f"错误：文件不存在 {image_path}"
                
            image = cv2.imread(image_path)
            
        if image is None:
            return f"错误：无法读取图像文件"
        
        # 获取图像基本信息
        height, width, channels = image.shape
        
        # 计算图像统计信息
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # 提取边缘用于结构分析
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        edge_ratio = edge_pixels / (height * width)
        
        analysis_result = f"""
卫星图像分析结果:
- 图像尺寸: {width}x{height} 像素
- 通道数: {channels}
- 平均亮度: {mean_intensity:.2f}
- 亮度标准差: {std_intensity:.2f}
- 边缘像素比例: {edge_ratio:.2%}
- 结构复杂度: {'高' if edge_ratio > 0.1 else '中' if edge_ratio > 0.05 else '低'}
"""
        return analysis_result
        
    except Exception as e:
        return f"图像分析失败: {str(e)}" 