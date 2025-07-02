from typing import List, Dict, Any, Optional
from dashscope import MultiModalConversation
from loguru import logger
import json
import asyncio


class LVM:
    """多模态大语言模型类，支持图像和文本输入
    
    Args:
        api_key (str): API密钥
        model (str): 模型名称，默认使用qwen-vl-plus-latest
        max_tokens (int): 最大token数
    """
    
    def __init__(self, 
                 api_key: str,
                 model: str = "qwen-vl-plus-latest",
                 max_tokens: int = 4000):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens # 控制的是输出的最大token长度
        
    async def chat(self, 
                   messages: List[Dict[str, Any]], 
                   tools: Optional[List[Dict]] = None,
                   tool_choice: str = "auto") -> Dict[str, Any]:
        """多模态对话
        
        Args:
            messages: 消息列表，支持文本和图像
            tools: 工具列表
            tool_choice: 工具选择策略
            
        Returns:
            模型响应
        """
        try:
            # 调试：检查传入的消息格式
            logger.debug(f"传入的消息类型: {type(messages)}")
            if messages is not None and hasattr(messages, '__iter__'):
                try:
                    logger.debug(f"消息长度: {len(messages) if hasattr(messages, '__len__') else 'Unknown'}")
                    for i, msg in enumerate(messages):
                        if msg is not None:
                            logger.debug(f"消息 {i}: type={type(msg)}, keys={list(msg.keys()) if isinstance(msg, dict) else 'Not dict'}")
                        else:
                            logger.debug(f"消息 {i}: None")
                except Exception as debug_e:
                    logger.error(f"调试消息时出错: {debug_e}")
            else:
                logger.error(f"messages为None或不可迭代: {messages}")
            
            # 清理消息，确保content不为None
            clean_messages = []
            for msg in messages:
                if msg and isinstance(msg, dict):
                    clean_msg = msg.copy()
                    if clean_msg.get("content") is None:
                        clean_msg["content"] = ""
                    # 处理role字段缺失的情况
                    if "role" not in clean_msg:
                        logger.warning("消息缺少role字段，跳过该消息")
                        continue
                    clean_messages.append(clean_msg)
            
            # 构建API调用参数
            call_params = {
                "model": self.model,
                "messages": clean_messages,
                "api_key": self.api_key,
                "max_tokens": self.max_tokens
            }
            
            # 只在有工具时添加tools参数
            if tools and len(tools) > 0:
                call_params["tools"] = tools
            
            logger.debug(f"API调用参数: {list(call_params.keys())}")
            
            # 在线程池中调用DashScope的多模态对话接口（因为它是同步API）
            response = await asyncio.to_thread(
                MultiModalConversation.call,
                **call_params
            )
            
            # 调试：输出响应信息
            logger.debug(f"API响应: {response}")
            
            # 处理响应 - 安全访问避免KeyError
            try:
                if 'output' in response and 'choices' in response['output']:
                    choices = response['output']['choices']
                else:
                    logger.error("响应中没有output.choices")
                    return {
                        "content": "抱歉，模型响应格式错误",
                        "role": "assistant",
                        "tool_calls": []
                    }
            except (KeyError, TypeError):
                logger.error("无法访问response.output.choices")
                return {
                    "content": "抱歉，模型响应格式错误",
                    "role": "assistant",
                    "tool_calls": []
                }
            
            if not choices or len(choices) == 0:
                logger.error("API响应中没有choices")
                return {
                    "content": "抱歉，模型响应格式错误",
                    "role": "assistant",
                    "tool_calls": []
                }
            
            choice = choices[0]
            
            # 安全获取message
            try:
                if 'message' in choice:
                    message = choice['message']
                else:
                    logger.error("choice中没有message字段")
                    return {
                        "content": "抱歉，模型响应格式错误", 
                        "role": "assistant",
                        "tool_calls": []
                    }
            except (KeyError, TypeError):
                logger.error("无法访问choice.message")
                return {
                    "content": "抱歉，模型响应格式错误",
                    "role": "assistant", 
                    "tool_calls": []
                }
            
            # 构建统一的响应格式 - 安全访问避免KeyError
            content = None
            content_data = None
            
            try:
                # DashScope对象支持字典式访问
                if 'content' in message:
                    content_data = message['content']
            except (KeyError, TypeError):
                content_data = None
            
            if content_data is not None:
                if isinstance(content_data, list):
                    if len(content_data) > 0:
                        if isinstance(content_data[0], dict) and "text" in content_data[0]:
                            content = content_data[0]["text"]
                        else:
                            content = str(content_data[0])
                    else:
                        # 如果是空列表，设置默认内容而不是None
                        content = ""
                elif isinstance(content_data, str):
                    content = content_data
                else:
                    content = str(content_data)
            else:
                # 如果content_data为None，设置默认内容
                content = ""
            
            result = {
                "content": content,
                "role": "assistant", 
                "tool_calls": []
            }
            
            # 处理工具调用 - 安全检查避免KeyError
            tool_calls_data = None
            try:
                # 直接尝试访问，DashScope对象支持字典式访问
                if 'tool_calls' in message:
                    tool_calls_data = message['tool_calls']
            except (KeyError, TypeError):
                # 如果不是字典类型或没有该键
                tool_calls_data = None
            
            if tool_calls_data is not None:
                # 确保 tool_calls 是可迭代的
                tool_calls = tool_calls_data if isinstance(tool_calls_data, (list, tuple)) else []
                logger.debug(f"处理 {len(tool_calls)} 个工具调用")
                for i, tool_call in enumerate(tool_calls):
                    logger.debug(f"工具调用 {i}: type={type(tool_call)}, data={tool_call}")
                    # 兼容字典和对象两种格式
                    if isinstance(tool_call, dict):
                        result["tool_calls"].append({
                            "id": tool_call.get("id"),
                            "function": {
                                "name": tool_call.get("function", {}).get("name"),
                                "arguments": tool_call.get("function", {}).get("arguments")
                            }
                        })
                    else:
                        # 对象格式（备用）
                        result["tool_calls"].append({
                            "id": getattr(tool_call, 'id', None),
                            "function": {
                                "name": getattr(tool_call.function, 'name', None) if hasattr(tool_call, 'function') else None,
                                "arguments": getattr(tool_call.function, 'arguments', None) if hasattr(tool_call, 'function') else None
                            }
                        })
            
            # 打印响应内容
            if result["content"]:
                logger.info(f"LVM响应: {result['content']}")
                
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"LVM调用失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return {
                "content": f"调用失败: {str(e)}",
                "role": "assistant", 
                "tool_calls": []
            } 