import os
import glob
import re

def fix_numpy_issues():
    # 修正対象のファイルを探す（coreフォルダ内と実行ファイル）
    files = glob.glob('core/*.py') + ['run_estimation.py']
    
    # 廃止された型を新しい型に置換する正規表現パターン
    # np.int32などは巻き込まないように「後ろに数字がないもの」だけを対象にします
    patterns = [
        (r'np\.int(?![0-9])', 'int'),       # np.int -> int
        (r'np\.float(?![0-9])', 'float'),   # np.float -> float
        (r'np\.bool(?![0-9])', 'bool'),     # np.bool -> bool
        (r'np\.object(?![0-9])', 'object'), # np.object -> object
    ]
    
    print("=== NumPy互換性修正を開始します ===")
    for filepath in files:
        if not os.path.exists(filepath): continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        for pat, repl in patterns:
            content = re.sub(pat, repl, content)
            
        if content != original_content:
            print(f"修正しました: {filepath}")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            print(f"変更なし: {filepath}")

    print("=== 完了しました ===")

if __name__ == "__main__":
    fix_numpy_issues()