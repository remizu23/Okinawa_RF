import torch
import matplotlib.pyplot as plt

# ==========================================
# ここに .pth ファイルのパスを指定してください
model_path = "/home/mizutani/projects/RF/runs/20251216_225748/model_weights_20251216_225748.pth"
# ==========================================

def plot_loss_from_pth(pth_path):
    try:
        # CPUで読み込む設定（GPUがなくても動くように）
        checkpoint = torch.load(pth_path, weights_only=False, map_location=torch.device('cpu'), )
        
        # historyデータの取得
        if 'history' not in checkpoint:
            print("エラー: このモデルファイルには 'history' データが含まれていません。")
            return

        history = checkpoint['history']
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])

        # グラフの描画
        epochs = range(1, len(train_loss) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label='Training Loss', marker='o')
        if val_loss:
            plt.plot(epochs, val_loss, label='Validation Loss', marker='x')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 画像として保存または表示
        plt.savefig("loss_graph.png")
        print("グラフを 'loss_graph.png' として保存しました。")
        plt.show()

    except FileNotFoundError:
        print(f"ファイルが見つかりません: {pth_path}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    plot_loss_from_pth(model_path)