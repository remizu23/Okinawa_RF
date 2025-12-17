import random

def show_comparison(teacher_df, result_df, num_samples=5):
    """
    ランダムにサンプルを選んで、正解と予測を並べて表示する
    """
    indices = random.sample(range(len(teacher_df)), num_samples)
    
    print(f"\n=== Qualitative Comparison (Random {num_samples} samples) ===")
    
    for idx in indices:
        gt_seq = clean_sequence(teacher_df.iloc[idx].values)
        pred_seq = clean_sequence(result_df.iloc[idx].values)
        
        dist = edit_distance(gt_seq, pred_seq)
        
        print(f"\nSample ID: {idx}")
        print(f"  Truth: {gt_seq}")
        print(f"  Pred : {pred_seq}")
        print(f"  Diff : {dist} (Length: T={len(gt_seq)}, P={len(pred_seq)})")
        
        # 完全一致かどうか
        if gt_seq == pred_seq:
            print("  Result:PERFECT MATCH! 🎉")
        else:
            # 長さが極端に違うかチェック
            if len(pred_seq) < len(gt_seq) * 0.5:
                print("  Result: Too Short (Early Stopping?)")
            elif len(pred_seq) > len(gt_seq) * 1.5:
                print("  Result: Too Long (Looping?)")
            else:
                print("  Result: Mismatch")

# 実行
show_comparison(teacher_df, result_df, num_samples=10)