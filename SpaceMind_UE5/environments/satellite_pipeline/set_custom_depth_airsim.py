"""
在AirSim运行时设置Custom Depth Stencil值
按照基本定义.md中的部件分割标准

使用方法：
1. 在UE模式， Output Log → Python 中运行以下代码：
import unreal
actors = unreal.EditorLevelLibrary.get_all_level_actors()
for a in actors:
    if a.get_class().get_name() == "StaticMeshActor":
        print(f"{a.get_name()}|{a.get_actor_label()}")

2. 复制UE输出粘贴到 NAME_LABEL_MAPPING（LogPython:前缀会自动去除）
3. 运行方式：
   - 切换到Airsim模式后，独立运行：python set_custom_depth_airsim.py
"""
import airsim
import time
import re
import argparse
from pathlib import Path
from collections import defaultdict

# ============================================================
# 👇 在这里粘贴从UE复制的映射（支持直接粘贴，LogPython:前缀会自动去除）
# ============================================================
NAME_LABEL_MAPPING = """
LogPython: StaticMeshActor_0|earth
LogPython: StaticMeshActor_5|moon
LogPython: StaticMeshActor_13|MilkyWay
LogPython: StaticMeshActor_978|CAPSTONE_CAPSTONE_main_body1
LogPython: StaticMeshActor_979|CAPSTONE_CAPSTONE_payload1
LogPython: StaticMeshActor_980|CAPSTONE_CAPSTONE_solar_panel1
LogPython: StaticMeshActor_981|CAPSTONE_CAPSTONE_solar_panel2
LogPython: StaticMeshActor_982|CAPSTONE_CAPSTONE_thruster1
"""
# ============================================================

# 基本定义.md中的渲染通道值映射
CATEGORY_STENCIL_VALUES = {
    'main_body': [13, 15, 16, 19],
    'solar_panel': [20, 23, 26, 27],
    'dish_antenna': [30, 35, 38, 39],
    'omni_antenna': [40, 44, 48, 49],
    'payload': [50, 51, 53, 58],
    'thruster': [61, 63, 66, 69],
    'adapter_ring': [74, 77],
}

# 关键字匹配规则
CATEGORY_KEYWORDS = {
    'main_body': ['main_body', 'body', 'mainbody'],
    'solar_panel': ['solar_panel', 'solar', 'panel'],
    'dish_antenna': ['dish', 'dish_antenna'],
    'omni_antenna': ['omni', 'antenna', 'omni_antenna'],
    'payload': ['payload'],
    'thruster': ['thruster', 'engine', 'nozzle'],
    'adapter_ring': ['adapter', 'ring', 'adapter_ring'],
}

# 背景物体（不设置CustomDepth）
# 注意：不包含'star'，避免误判卫星名称中包含star的情况（如Double_Star）
BACKGROUND_KEYWORDS = ['earth', 'moon', 'milkyway']

def parse_mapping():
    """解析NAME_LABEL_MAPPING（兼容LogPython:前缀）"""
    mapping = {}
    for line in NAME_LABEL_MAPPING.strip().split('\n'):
        line = line.strip()
        
        # 移除 LogPython: 前缀（如果有）
        if line.startswith('LogPython:'):
            line = line.replace('LogPython:', '').strip()
        
        if '|' in line and not line.startswith('#'):
            parts = line.split('|')
            if len(parts) == 2:
                mapping[parts[0].strip()] = parts[1].strip()
    return mapping


def load_mapping_text(mapping_file):
    """从文件加载映射文本"""
    if not mapping_file:
        return None
    p = Path(mapping_file)
    if not p.exists():
        raise FileNotFoundError(f"mapping_file 不存在: {p}")
    return p.read_text(encoding="utf-8")


def classify_component(label):
    """根据label自动分类并返回优先级
    
    注意：优先匹配卫星部件，避免被背景关键词误判
    """
    label_lower = label.lower()
    
    # 先检查是否是卫星部件（优先级最高）
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in label_lower:
                # 提取序号用于排序
                match = re.search(r'(\d+)$', label)
                priority = int(match.group(1)) if match else 0
                return category, priority
    
    # 再检查是否是背景
    for bg in BACKGROUND_KEYWORDS:
        if bg in label_lower:
            return 'background', 0
    
    return 'unknown', 0


def generate_config():
    """自动生成配置"""
    mapping = parse_mapping()
    
    # 按类别分组
    categorized = defaultdict(list)
    for name, label in mapping.items():
        category, priority = classify_component(label)
        categorized[category].append((name, label, priority))
    
    config = {}
    
    # 背景物体
    for name, label, _ in categorized.get('background', []):
        config[name] = (False, 0, f"{label} - 背景")
    
    # 其他类别：按priority排序分配stencil值
    category_cn = {
        'main_body': '主体',
        'solar_panel': '太阳能板',
        'dish_antenna': '高增益天线',
        'omni_antenna': '低增益天线',
        'payload': '载荷',
        'thruster': '推进器',
        'adapter_ring': '对接环',
    }
    
    for category, items in categorized.items():
        if category in ['background', 'unknown']:
            continue
        
        items.sort(key=lambda x: x[2])  # 按priority排序
        stencil_values = CATEGORY_STENCIL_VALUES.get(category, [100])
        
        for i, (name, label, _) in enumerate(items):
            stencil = stencil_values[i % len(stencil_values)]
            config[name] = (True, stencil, f"{label} - {category_cn.get(category, category)}")
    
    # 未知类别
    for name, label, _ in categorized.get('unknown', []):
        config[name] = (True, 255, f"{label} - 未分类")
    
    return config


def set_custom_depth_for_all_objects(client):
    """设置所有物体的Custom Depth和Stencil值"""
    
    # 自动生成配置
    config = generate_config()
    
    print("\n" + "="*70)
    print("自动生成的配置：")
    print("="*70)
    
    # 打印配置摘要
    stencil_groups = defaultdict(list)
    for name, (enable, stencil, desc) in config.items():
        if enable:
            stencil_groups[stencil].append(desc.split(' - ')[0])
    
    for stencil in sorted(stencil_groups.keys()):
        items = stencil_groups[stencil]
        print(f"  Stencil {stencil}: {', '.join(items)}")
    
    print("\n" + "="*70)
    print("开始设置Custom Depth Stencil值...")
    print("="*70)
    
    success_count = 0
    failed_count = 0
    
    for actor_name, (enable, stencil, desc) in config.items():
        if not enable:
            # 对于不需要的物体，设置为0（背景）
            # 注意：AirSim的simSetSegmentationObjectID主要用于启用分割
            # 关闭可能需要在UE端处理，这里尝试设为0
            try:
                client.simSetSegmentationObjectID(actor_name, 0)
                print(f"  ✓ {desc} - 已禁用")
                success_count += 1
            except Exception as e:
                print(f"  ✗ {desc} - 失败: {e}")
                failed_count += 1
        else:
            # 启用CustomDepth并设置Stencil值
            try:
                success = client.simSetSegmentationObjectID(actor_name, stencil)
                if success:
                    print(f"  ✓ {desc} -> Stencil={stencil}")
                    success_count += 1
                else:
                    print(f"  ✗ {desc} -> Stencil={stencil} (返回False)")
                    failed_count += 1
            except Exception as e:
                print(f"  ✗ {desc} - 异常: {e}")
                failed_count += 1
    
    print("="*70)
    print(f"设置完成: 成功 {success_count} 个, 失败 {failed_count} 个")
    print("="*70)
    
    return success_count, failed_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_file", type=str, default="", help="从文件读取 name|label 映射（替代手动粘贴 NAME_LABEL_MAPPING）")
    args = parser.parse_args()

    global NAME_LABEL_MAPPING
    if args.mapping_file:
        NAME_LABEL_MAPPING = load_mapping_text(args.mapping_file) or ""

    # 连接AirSim
    print("连接AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("连接成功！")
    
    # 等待AirSim完全初始化
    print("\n等待AirSim初始化...")
    time.sleep(2)
    
    # 设置Custom Depth
    success, failed = set_custom_depth_for_all_objects(client)
    
    print(f"\n💡 提示:")
    print(f"1. 已按照基本定义.md中的部件分割标准，自动分配了渲染自定义深度模板值")
    print(f"2. 请手动在细节搜索深度，检查分配值是否正确")
    print(f"3. 更换卫星时，只需修改顶部的 NAME_LABEL_MAPPING 即可")


if __name__ == "__main__":
    main()

