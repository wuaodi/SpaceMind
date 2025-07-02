import sys
import os
from pathlib import Path

# 使用更规范的路径处理方式
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from prompt import SYSTEM_PROMPT
from agent import VisionAgent, ToolManager, MemoryManager, LVM
from tool import *
from loguru import logger
import asyncio


async def main():
    # 获取API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("未检测到DASHSCOPE_API_KEY环境变量")
        api_key = input("请输入通义千问API密钥: ").strip()
        if not api_key:
            print("错误：必须提供API密钥")
            return
    
    # 初始化多模态大模型
    try:
        lvm = LVM(api_key=api_key, model="qwen-vl-plus-latest")
    except Exception as e:
        logger.error(f"初始化LVM失败: {e}")
        return
    
    # 初始化管理器
    tool_manager = ToolManager()
    memory_manager = MemoryManager(max_memory=5)  # 减少记忆长度，避免上下文过长
    
    # 初始化智能体
    agent = VisionAgent(
        lvm=lvm,
        tool_manager=tool_manager,
        memory_manager=memory_manager,
        max_step=3  # 减少最大步数，降低API调用频率
    )
    
    # 注册工具
    agent.add_tool(extract_edges, "extract_edges")
    agent.add_tool(analyze_satellite_image, "analyze_satellite_image")
    agent.add_tool(terminate, "terminate")
    
    print("空间在轨服务智能体已启动！")
    print("支持的功能：")
    print("1. 图像边缘提取")
    print("2. 卫星图像分析")
    print("请提供图像路径和任务描述，输入quit退出")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请描述任务（可包含图像路径）: ").strip()
            if user_input.lower() == "quit":
                break
            
            if not user_input:
                print("请输入任务描述")
                continue
            
            # 构建消息
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]
            
            # 处理图像输入
            image_extensions = {'.jpg', '.jpeg', '.png'}
            
            # 查找用户输入中的图像路径
            for word in user_input.split():
                if any(word.lower().endswith(ext) for ext in image_extensions):
                    image_path = Path(word)
                    if image_path.exists():
                        try:
                            import base64
                            with open(image_path, "rb") as image_file:
                                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                            
                            # 确定图像格式
                            suffix = image_path.suffix.lower()
                            image_format = "png" if suffix == '.png' else "jpeg"
                            
                            # 使用DashScope官方格式
                            messages[-1]["content"] = [
                                {
                                    "type": "image",
                                    "image": f"data:image/{image_format};base64,{base64_image}"
                                },
                                {"type": "text", "text": user_input}
                            ]
                            logger.info(f"成功加载图像: {image_path}")
                            break
                        except Exception as img_e:
                            logger.error(f"加载图像失败 {image_path}: {img_e}")
                            print(f"警告：无法加载图像 {image_path}，将仅处理文本内容")
                    else:
                        print(f"警告：图像文件不存在: {word}")
            
            # 运行智能体，添加超时控制
            logger.info("智能体开始处理任务...")
            try:
                # 设置30秒超时
                await asyncio.wait_for(agent.run(messages), timeout=30.0)
            except asyncio.TimeoutError:
                logger.error("智能体执行超时（30秒），请检查网络连接或减少任务复杂度")
            except asyncio.CancelledError:
                logger.warning("智能体执行被取消")
            except Exception as agent_e:
                logger.error(f"智能体执行失败: {agent_e}")
                # 继续下一轮，不退出程序
            
        except KeyboardInterrupt:
            print("\n检测到键盘中断")
            break
        except Exception as e:
            logger.error(f"发生错误: {e}")
            print("发生错误，请重试")
    
    print("\n感谢使用空间在轨服务智能体！")


if __name__ == "__main__":
    # 设置日志级别
    logger.add("lvm_agent.log", rotation="10 MB", level="INFO")
    
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"程序异常退出: {e}")
        print(f"程序异常退出: {e}") 