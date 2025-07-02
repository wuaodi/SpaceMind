"""视觉消息处理辅助模块"""

import base64
import os
import tempfile
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger


def extract_image_from_message(message: Dict) -> Optional[str]:
    """从消息中提取图像数据
    
    Args:
        message: 消息字典
        
    Returns:
        图像的base64数据URL或None
    """
    content = message.get("content", "")
    
    # 如果content是列表（多模态消息）
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                image_data = item.get("image", "")
                if image_data:
                    return image_data
    
    return None


def save_image_temporarily(base64_data: str) -> Optional[str]:
    """将base64图像数据保存为临时文件
    
    Args:
        base64_data: base64编码的图像数据（可能包含data URL前缀）
        
    Returns:
        临时文件路径或None
    """
    try:
        # 提取纯base64数据
        if base64_data.startswith("data:image"):
            # 从data URL中提取base64部分
            base64_data = base64_data.split(",")[1]
        
        # 解码base64数据
        image_data = base64.b64decode(base64_data)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(image_data)
            temp_path = tmp_file.name
        
        logger.info(f"图像已保存到临时文件: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"保存图像失败: {e}")
        return None


def find_image_in_messages(messages: List[Dict]) -> Optional[str]:
    """在消息列表中查找最近的图像
    
    Args:
        messages: 消息列表
        
    Returns:
        图像的base64数据URL或None
    """
    # 从最新的消息开始查找
    for message in reversed(messages):
        image_data = extract_image_from_message(message)
        if image_data:
            return image_data
    
    return None


def inject_image_path_to_tools(messages: List[Dict]) -> Tuple[List[Dict], Optional[str]]:
    """为消息列表注入图像路径信息
    
    如果消息中包含图像，将其保存为临时文件，并在最后添加一条包含路径信息的用户消息
    
    Args:
        messages: 原始消息列表
        
    Returns:
        (处理后的消息列表, 临时文件路径或None)
    """
    # 查找图像
    image_data = find_image_in_messages(messages)
    
    if not image_data:
        return messages, None
    
    # 保存为临时文件
    temp_path = save_image_temporarily(image_data)
    
    if not temp_path:
        return messages, None
    
    # 创建消息副本
    messages_copy = messages.copy()
    
    # 添加一条隐含的系统消息，告知模型图像路径
    messages_copy.append({
        "role": "system",
        "content": f"[图像已自动保存到临时文件: {temp_path}，请在调用工具时使用此路径作为image_path参数]"
    })
    
    return messages_copy, temp_path


def cleanup_temp_image(temp_path: Optional[str]):
    """清理临时图像文件
    
    Args:
        temp_path: 临时文件路径
    """
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
            logger.info(f"已清理临时文件: {temp_path}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}") 