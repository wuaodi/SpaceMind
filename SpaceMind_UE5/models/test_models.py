#!/usr/bin/env python3
"""
快速测试模型配置是否可用
运行方式: python test_models.py --model qwen3-vl-235b
"""

import asyncio
import argparse
from config.cli_config import AVAILABLE_MODELS, Config
from models.model_client import ModelClient


def get_model_env_source(model_name: str) -> str:
    if model_name in {"qwen3-vl-235b", "qwen3-vl-8b", "qwen3.5-397b", "qwen3.5-27b", "kimi-k2.5"}:
        return "MODELSCOPE_API_KEY"
    if model_name in {"gpt-4.1", "gpt-4o", "claude-sonnet-4"}:
        return "METACHAT_API_KEY"
    return "unknown"

async def test_model(model_name: str):
    """测试单个模型是否可用"""
    if model_name not in AVAILABLE_MODELS:
        print(f"❌ 模型 '{model_name}' 不在可用列表中")
        print(f"可用模型: {', '.join(AVAILABLE_MODELS.keys())}")
        return False
    
    cfg = AVAILABLE_MODELS[model_name]
    print(f"\n🔍 测试模型: {model_name}")
    print(f"   API: {cfg['base_url']}")
    print(f"   Model name: {cfg['name']}")
    print(f"   Env key: {get_model_env_source(model_name)}")
    print(f"   API key: {'已配置' if cfg['api_key'] else '❌ 未配置'}...")
    
    if not cfg['api_key']:
        print(f"❌ API key 未配置，请检查 .env 文件")
        return False
    
    try:
        client = ModelClient(
            api_key=cfg['api_key'],
            base_url=cfg['base_url'],
            model_name=cfg['name'],
            max_tokens=cfg['max_tokens'],
            temperature=cfg['temperature'],
        )
        
        # 简单测试：发送一个纯文本请求
        print("   发送测试请求...")
        msg = await client.chat_text_only(
            system_prompt="You are a helpful assistant.",
            user_text="Say 'Hello' in one word.",
        )
        
        if msg and msg.content:
            print(f"✅ 模型可用！响应: {msg.content[:50]}...")
            return True
        else:
            print("❌ 模型响应为空")
            if client.last_error:
                print(f"   详细错误: {client.last_error}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

async def test_all_models():
    """测试所有模型"""
    print("=" * 60)
    print("测试所有模型配置...")
    print("=" * 60)
    
    results = {}
    for model_name in AVAILABLE_MODELS.keys():
        results[model_name] = await test_model(model_name)
        await asyncio.sleep(0.5)  # 避免请求过快
    
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    for model_name, success in results.items():
        status = "✅ 可用" if success else "❌ 不可用"
        print(f"  {model_name:20s} {status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="测试指定模型（不指定则测试所有）")
    args = parser.parse_args()
    
    Config.parse_args()  # 加载环境变量
    
    if args.model:
        asyncio.run(test_model(args.model))
    else:
        asyncio.run(test_all_models())
