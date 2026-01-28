import torch
import numpy as np
import os
import glob
from KP_RF import KoopmanRoutesFormer

# 設定
# ★最新のモデルパスを指定してください
MODEL_PATH = '/home/mizutani/projects/RF/runs/20260121_145835/model_weights_20260121_145835.pth' 
DEVICE = "cpu"

def main():
    print(f"Loading weights from: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    
    # state_dictの取得
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    # delta_bin_weight を探す
    if "delta_bin_weight" in state:
        weights = state["delta_bin_weight"].numpy() # Shape: [2, 3]
        
        print("\n=== Learned Bias Weights (delta_bin_weight) ===")
        print("Shape:", weights.shape)
        print("Rows: [0: Inactive(2024), 1: Active(2025)]")
        print("Cols: [0: Toward, 1: Same, 2: Away]")
        print("-" * 40)
        
        # 数値を表示
        categories = ["Toward (接近)", "Same   (維持)", "Away   (離脱)"]
        
        w_inactive = weights[0]
        w_active = weights[1]
        
        print(f"{'Category':<15} | {'Inactive (OFF)':<15} | {'Active (ON)':<15} | {'Diff (ON-OFF)':<15}")
        print("-" * 70)
        
        for i, cat in enumerate(categories):
            off_val = w_inactive[i]
            on_val  = w_active[i]
            diff    = on_val - off_val
            print(f"{cat:<15} | {off_val: .4f}          | {on_val: .4f}        | {diff: .4f}")
            
        print("-" * 70)
        print("\n【解釈のヒント】")
        print("・Diffが「プラス」の項目は、広場イベント時にモデルが推奨している行動です。")
        print("・もし Same の Diff が大きくプラスなら、やはり「滞留」を学習しています。")
        print("・Away の Diff がマイナスなら、「流出阻止」を学習しています。")
        
    else:
        print("Error: 'delta_bin_weight' not found in the checkpoint.")

if __name__ == "__main__":
    main()