from typing import List, Dict, Optional
import json
from .memory_manager import MemoryManager
from .tool_manager import ToolManager
from .lvm import LVM
from .vision_helper import inject_image_path_to_tools, cleanup_temp_image
from loguru import logger
from prompt import NEXT_STEP_PROMPT, FINAL_STEP_PROMPT


class VisionAgent:
    """视觉智能体，专门处理空间在轨服务任务
    
    Args:
        lvm: 多模态大模型
        tool_manager: 工具管理器
        memory_manager: 记忆管理器
        max_step: 最大执行步数
    """
    
    def __init__(self, 
                 lvm: LVM,
                 tool_manager: ToolManager,
                 memory_manager: MemoryManager,
                 max_step: int = 5):
        self.lvm = lvm
        self.tool_manager = tool_manager
        self.memory_manager = memory_manager
        self.max_step = max_step
        self.temp_image_path = None  # 跟踪临时图像文件
        
    async def think(self, messages: List[Dict]) -> bool:
        """思考下一步操作"""
        # 创建消息副本，避免修改原始列表
        messages_copy = []
        if messages is not None:
            for msg in messages:
                if msg and isinstance(msg, dict) and "role" in msg:
                    # 确保content字段存在
                    if "content" not in msg:
                        msg["content"] = ""
                    messages_copy.append(msg)
        else:
            logger.error("传入的messages为None")
            return False
        
        # 检查是否有有效消息
        if not messages_copy:
            logger.error("没有有效的消息")
            return False
        
        # 处理消息中的图像，注入临时文件路径
        processed_messages, temp_path = inject_image_path_to_tools(messages_copy)
        if temp_path:
            self.temp_image_path = temp_path
            logger.info(f"检测到图像，已保存到临时文件: {temp_path}")
        
        processed_messages.append({"role": "user", "content": NEXT_STEP_PROMPT})
        
        logger.debug(f"发送给LVM的消息数量: {len(processed_messages)}")
        
        response = await self.lvm.chat(
            messages=processed_messages,
            tools=self.tool_manager.get_tool_schema_list()
        )
        
        # 检查API是否返回错误消息
        if response.get("content") == "抱歉，模型响应格式错误":
            raise Exception("LVM API调用失败：模型响应格式错误")
        
        self.memory_manager.add_message(response)
        
        if response.get("content"):
            logger.info(f"智能体思考: {response['content']}")
            
        return bool(response.get("tool_calls"))
    
    async def act(self, message: Dict) -> bool:
        """执行工具调用"""
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            logger.warning("没有工具调用")
            return False
            
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                logger.warning(f"无效的工具调用格式: {tool_call}")
                continue
                
            function_info = tool_call.get("function", {})
            tool_name = function_info.get("name")
            tool_args = function_info.get("arguments", "{}")
            
            if not tool_name:
                logger.warning("工具调用缺少工具名称")
                continue
            
            logger.info(f"调用工具: {tool_name}")
            
            try:
                # 解析参数
                if tool_args and tool_args != "{}":
                    # 处理可能包含多个函数调用的情况
                    # 只提取第一个有效的JSON
                    tool_args = tool_args.strip()
                    
                    # 如果参数中包含特殊标记或换行符，说明LVM试图调用多个函数
                    if "✿FUNCTION✿" in tool_args or "\n" in tool_args:
                        # 尝试提取第一行的JSON
                        first_line = tool_args.split('\n')[0].strip()
                        if first_line:
                            tool_args = first_line
                        logger.warning("检测到LVM尝试在一个响应中调用多个函数，只处理第一个")
                    
                    tool_args = json.loads(tool_args)
                    result = await self.tool_manager.execute_tool(tool_name, **tool_args)
                else:
                    result = await self.tool_manager.execute_tool(tool_name)
                
                # 记录工具结果，限制长度避免API问题
                result_str = str(result)
                if len(result_str) > 10000:  # 限制结果长度
                    result_str = result_str[:10000] + "..."
                
                tool_message = {
                    "role": "tool",
                    "content": result_str,
                    "tool_call_id": tool_call.get("id", "unknown")
                }
                self.memory_manager.add_message(tool_message)
                logger.info(f"工具 {tool_name} 执行成功，结果: {str(result)[:100]}...")
                
                if tool_name == "terminate":
                    return True
                    
            except Exception as e:
                logger.error(f"工具执行失败: {e}")
                
        return False
    
    async def run(self, messages: List[Dict]):
        """运行智能体"""
        # 添加初始消息到记忆
        for msg in messages:
            self.memory_manager.add_message(msg)
        
        step = 0
        terminated = False
        
        while step < self.max_step and not terminated:
            logger.info(f"执行第 {step + 1} 步")
            
            # 思考
            should_act = await self.think(self.memory_manager.get_memory())
            
            if should_act:
                # 行动
                last_message = self.memory_manager.get_memory()[-1]
                terminated = await self.act(last_message)
            
            step += 1
        
        # 生成最终总结
        if terminated:
            try:
                # 提取关键信息用于总结（避免图像数据导致token过多）
                memory = self.memory_manager.get_memory()
                
                # 构建简洁的总结信息
                summary_info = []
                tools_used = []
                results = []
                
                for msg in memory:
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        # 收集使用的工具
                        for tool_call in msg.get("tool_calls", []):
                            tool_name = tool_call.get("function", {}).get("name")
                            if tool_name:
                                tools_used.append(tool_name)
                    elif msg.get("role") == "tool":
                        # 收集工具执行结果
                        content = msg.get("content", "")
                        if len(content) > 200:
                            content = content[:200] + "..."
                        results.append(content)
                
                # 构建总结消息
                summary_messages = [
                    {"role": "system", "content": "你是一个专业的空间在轨服务AI助手，请根据提供的信息生成任务总结。"},
                    {
                        "role": "user", 
                        "content": f"""请总结以下任务执行情况：

使用的工具: {', '.join(set(tools_used))}

工具执行结果:
{chr(10).join([f"- {result}" for result in results])}

请提供一个专业的总结，包括：
1. 执行了哪些主要操作
2. 获得的关键结果和发现
3. 对空间在轨服务的建议"""
                    }
                ]
                
                final_response = await self.lvm.chat(
                    messages=summary_messages,
                    tools=None  # 不使用工具
                )
                
                # 检查API是否返回错误消息  
                if final_response.get("content") == "抱歉，模型响应格式错误":
                    logger.warning("最终总结API调用失败")
                elif final_response.get("content"):
                    logger.info(f"最终总结: {final_response['content']}")
                    print(f"\n📋 任务总结:\n{final_response['content']}")
                    
            except Exception as summary_e:
                logger.warning(f"最终总结生成失败: {summary_e}")
                # 不抛出异常，任务已经完成
        
        logger.info("任务完成")
        self.memory_manager.clear()
        
        # 清理临时图像文件
        if self.temp_image_path:
            cleanup_temp_image(self.temp_image_path)
            self.temp_image_path = None
        
    def add_tool(self, func, tool_name: Optional[str] = None):
        """添加工具"""
        self.tool_manager.register_tool(func, tool_name) 