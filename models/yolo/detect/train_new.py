#!/usr/bin/env python3
"""
Hyper-YOLO 训练脚本 - 使用 YOLO API 方式
"""

from ultralytics import YOLO
import yaml
import tempfile
import os
import sys
import argparse
import datetime  # 新增：用于生成时间戳

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='model config path')
    parser.add_argument('--data', type=str, required=True, help='data config path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--scale', type=str, default='l', choices=['n', 's', 'm', 'l', 'x'], help='model scale')
    parser.add_argument('--name', type=str, default=None, help='experiment name (default: auto-generated with timestamp)')
    parser.add_argument('--pretrained', type=bool, default=False, help='use pretrained weights')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--amp', type=bool, default=False, help='use AMP')
    parser.add_argument('--no-timestamp', action='store_true', help='disable timestamp in experiment name')
    parser.add_argument('--description', type=str, default='', help='additional description for experiment')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Hyper-YOLO 训练脚本 (YOLO API 版本)")
    print("="*60)
    
    # 从配置文件名中提取模型名称
    model_base = os.path.basename(args.cfg).replace('.yaml', '')
    
    # 生成实验名称
    if args.name is None:
        # 基础名称
        base_name = f"{model_base}-{args.scale}"
        
        # 添加描述（如果有）
        if args.description:
            base_name += f"-{args.description}"
        
        # 添加时间戳（除非禁用）
        if not args.no_timestamp:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.name = f"{base_name}-{timestamp}"
        else:
            # 如果不加时间戳，检查是否存在同名文件夹，自动添加序号
            base_path = f"/root/autodl-tmp/Hyper-YOLO-main/runs/{base_name}"
            if os.path.exists(base_path):
                counter = 1
                while os.path.exists(f"{base_path}-{counter}"):
                    counter += 1
                args.name = f"{base_name}-{counter}"
            else:
                args.name = base_name
        
        print(f"📝 自动生成实验名称: {args.name}")
    else:
        # 如果手动指定了名称，但没加时间戳且未禁用，可以自动添加
        if not args.no_timestamp and args.name.find(datetime.datetime.now().strftime("%Y%m%d")) == -1:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.name = f"{args.name}-{timestamp}"
            print(f"📝 添加时间戳到实验名称: {args.name}")
    
    # 1. 读取配置文件
    print(f"\n📁 读取配置文件: {args.cfg}")
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 2. 设置环境变量
    print(f"\n📊 使用模型规模: {args.scale}")
    scale_params = cfg['scales'][args.scale]
    print(f"   参数: depth={scale_params[0]}, width={scale_params[1]}, max_channels={scale_params[2]}, threshold={scale_params[3]}")
    
    # 设置threshold环境变量
    for k, v in cfg['scales'].items():
        if len(v) >= 4:
            os.environ[f'HYPER_THRESHOLD_{k.upper()}'] = str(v[3])
            if k == args.scale:
                print(f"   ✅ 环境变量 HYPER_THRESHOLD_{k.upper()} = {v[3]}")
    
    # 3. 修改配置文件，添加threshold参数到模型定义
    threshold = cfg['scales'][args.scale][3]
    modified = False
    
    for i, layer in enumerate(cfg['head']):
        if len(layer) >= 4 and layer[2] == 'HyperComputeModule':
            if len(layer) >= 4:
                if isinstance(layer[3], list):
                    layer[3].append(threshold)
                else:
                    layer[3] = [layer[3], threshold]
                print(f"   ✅ HyperComputeModule 添加 threshold={threshold}")
                modified = True
    
    # 4. 添加scale参数到配置文件
    cfg['scale'] = args.scale
    
    # 5. 保存修改后的配置到临时文件
    temp_cfg = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False).name
    with open(temp_cfg, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    print(f"\n📝 临时配置文件已保存: {temp_cfg}")
    
    # 6. 加载模型
    print("\n🚀 正在加载模型...")
    model = YOLO(temp_cfg)
    
    # 7. 开始训练
    print("\n🎯 开始训练...")
    print(f"   数据集: {args.data}")
    print(f"   epochs: {args.epochs}")
    print(f"   batch: {args.batch}")
    print(f"   实验名称: {args.name}")
    print("="*60)
    
    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            project='/root/autodl-tmp/Hyper-YOLO-main/runs',
            name=args.name,
            exist_ok=False,  # 改为False，避免覆盖已有实验
            pretrained=args.pretrained,
            optimizer=args.optimizer,
            lr0=args.lr0,
            amp=args.amp
        )
        
        print(f"\n✅ 训练完成！")
        print(f"   结果保存在: /root/autodl-tmp/Hyper-YOLO-main/runs/{args.name}")
        print(f"   最佳模型: /root/autodl-tmp/Hyper-YOLO-main/runs/{args.name}/weights/best.pt")
        
        # 保存实验参数到文件
        exp_info = {
            'cfg': args.cfg,
            'data': args.data,
            'scale': args.scale,
            'epochs': args.epochs,
            'batch': args.batch,
            'imgsz': args.imgsz,
            'device': args.device,
            'optimizer': args.optimizer,
            'lr0': args.lr0,
            'amp': args.amp,
            'description': args.description,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        info_path = f"/root/autodl-tmp/Hyper-YOLO-main/runs/{args.name}/experiment_info.yaml"
        with open(info_path, 'w') as f:
            yaml.dump(exp_info, f, default_flow_style=False)
        print(f"📋 实验信息已保存: {info_path}")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_cfg)
            print(f"🧹 临时文件已删除: {temp_cfg}")
        except:
            pass
    
    return 0

if __name__ == '__main__':
    sys.exit(main())