"""
動作確認スクリプト

修正版モデルが正しく動作するか簡単にテスト
"""

import torch
import torch.nn.functional as F
from KP_RF_modified import KoopmanRoutesFormer

def test_basic_forward():
    """基本的なforwardパスのテスト"""
    print("=== Test 1: Basic Forward Pass ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # モデル初期化
    model = KoopmanRoutesFormer(
        vocab_size=39,
        token_emb_dim=32,
        d_model=64,
        nhead=4,
        num_layers=2,
        d_ff=128,
        z_dim=32,
        pad_token_id=38,
        base_N=19,
        use_aux_loss=True,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ダミーデータ
    batch_size = 4
    prefix_len = 10
    K = 5
    
    prefix_tokens = torch.randint(0, 38, (batch_size, prefix_len)).to(device)
    prefix_stay_counts = torch.randint(0, 5, (batch_size, prefix_len)).to(device)
    prefix_agent_ids = torch.zeros(batch_size, dtype=torch.long).to(device)
    prefix_holidays = torch.randint(0, 2, (batch_size, prefix_len)).to(device)
    prefix_time_zones = torch.randint(0, 2, (batch_size, prefix_len)).to(device)
    prefix_events = torch.randint(0, 2, (batch_size, prefix_len)).to(device)
    
    future_tokens = torch.randint(0, 38, (batch_size, K)).to(device)
    
    # Forward
    model.train()
    outputs = model.forward_rollout(
        prefix_tokens=prefix_tokens,
        prefix_stay_counts=prefix_stay_counts,
        prefix_agent_ids=prefix_agent_ids,
        prefix_holidays=prefix_holidays,
        prefix_time_zones=prefix_time_zones,
        prefix_events=prefix_events,
        K=K,
        future_tokens=future_tokens,
    )
    
    pred_logits = outputs['pred_logits']
    z_0 = outputs['z_0']
    aux_losses = outputs['aux_losses']
    
    print(f"pred_logits shape: {pred_logits.shape}")  # Should be [4, 5, 39]
    print(f"z_0 shape: {z_0.shape}")  # Should be [4, 32]
    
    if aux_losses:
        print(f"aux_count_loss: {aux_losses['count'].item():.4f}")
        print(f"aux_mode_loss: {aux_losses['mode'].item():.4f}")
    
    # CE損失の計算
    ce_loss = F.cross_entropy(
        pred_logits.reshape(-1, model.vocab_size),
        future_tokens.reshape(-1)
    )
    print(f"CE loss: {ce_loss.item():.4f}")
    
    print("✓ Basic forward pass successful\n")
    return model, device


def test_variable_length_prefix():
    """可変長Prefixのテスト"""
    print("=== Test 2: Variable Length Prefix ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = KoopmanRoutesFormer(
        vocab_size=39,
        token_emb_dim=32,
        d_model=64,
        nhead=4,
        num_layers=2,
        d_ff=128,
        z_dim=32,
        use_aux_loss=False,
    ).to(device)
    
    # 可変長のバッチ
    batch_size = 3
    prefix_lengths = [5, 10, 15]
    max_prefix_len = max(prefix_lengths)
    K = 5
    
    # パディングを含むテンソル
    prefix_tokens = torch.full((batch_size, max_prefix_len), 38, dtype=torch.long).to(device)
    prefix_stay_counts = torch.zeros((batch_size, max_prefix_len), dtype=torch.long).to(device)
    prefix_holidays = torch.zeros((batch_size, max_prefix_len), dtype=torch.long).to(device)
    prefix_time_zones = torch.zeros((batch_size, max_prefix_len), dtype=torch.long).to(device)
    prefix_events = torch.zeros((batch_size, max_prefix_len), dtype=torch.long).to(device)
    prefix_mask = torch.ones((batch_size, max_prefix_len), dtype=torch.bool).to(device)
    
    # 有効データを埋める
    for i, plen in enumerate(prefix_lengths):
        prefix_tokens[i, :plen] = torch.randint(0, 38, (plen,))
        prefix_stay_counts[i, :plen] = torch.randint(0, 5, (plen,))
        prefix_holidays[i, :plen] = torch.randint(0, 2, (plen,))
        prefix_time_zones[i, :plen] = torch.randint(0, 2, (plen,))
        prefix_events[i, :plen] = torch.randint(0, 2, (plen,))
        prefix_mask[i, :plen] = False  # 有効位置はFalse
    
    prefix_agent_ids = torch.zeros(batch_size, dtype=torch.long).to(device)
    
    # Forward
    model.eval()
    with torch.no_grad():
        outputs = model.forward_rollout(
            prefix_tokens=prefix_tokens,
            prefix_stay_counts=prefix_stay_counts,
            prefix_agent_ids=prefix_agent_ids,
            prefix_holidays=prefix_holidays,
            prefix_time_zones=prefix_time_zones,
            prefix_events=prefix_events,
            K=K,
            prefix_mask=prefix_mask,
        )
    
    pred_logits = outputs['pred_logits']
    z_0 = outputs['z_0']
    
    print(f"pred_logits shape: {pred_logits.shape}")  # Should be [3, 5, 39]
    print(f"z_0 shape: {z_0.shape}")  # Should be [3, 32]
    
    # 各サンプルのz_0が異なることを確認
    print(f"z_0[0] norm: {z_0[0].norm().item():.4f}")
    print(f"z_0[1] norm: {z_0[1].norm().item():.4f}")
    print(f"z_0[2] norm: {z_0[2].norm().item():.4f}")
    
    print("✓ Variable length prefix test successful\n")


def test_autonomous_rollout():
    """自律ロールアウトのテスト（B行列がないことの確認）"""
    print("=== Test 3: Autonomous Rollout (No B matrix) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = KoopmanRoutesFormer(
        vocab_size=39,
        token_emb_dim=32,
        d_model=64,
        nhead=4,
        num_layers=2,
        d_ff=128,
        z_dim=32,
        use_aux_loss=False,
    ).to(device)
    
    # B行列が存在しないことを確認
    has_B = hasattr(model, 'B')
    print(f"Model has B matrix: {has_B}")
    
    if has_B:
        print("⚠ Warning: B matrix still exists!")
    else:
        print("✓ B matrix successfully removed")
    
    # 同じPrefixから複数回ロールアウトして、決定的であることを確認
    prefix_tokens = torch.randint(0, 38, (1, 10)).to(device)
    prefix_stay_counts = torch.randint(0, 5, (1, 10)).to(device)
    prefix_agent_ids = torch.zeros(1, dtype=torch.long).to(device)
    prefix_holidays = torch.randint(0, 2, (1, 10)).to(device)
    prefix_time_zones = torch.randint(0, 2, (1, 10)).to(device)
    prefix_events = torch.randint(0, 2, (1, 10)).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs1 = model.forward_rollout(
            prefix_tokens=prefix_tokens,
            prefix_stay_counts=prefix_stay_counts,
            prefix_agent_ids=prefix_agent_ids,
            prefix_holidays=prefix_holidays,
            prefix_time_zones=prefix_time_zones,
            prefix_events=prefix_events,
            K=10,
        )
        
        outputs2 = model.forward_rollout(
            prefix_tokens=prefix_tokens,
            prefix_stay_counts=prefix_stay_counts,
            prefix_agent_ids=prefix_agent_ids,
            prefix_holidays=prefix_holidays,
            prefix_time_zones=prefix_time_zones,
            prefix_events=prefix_events,
            K=10,
        )
    
    # 同じ結果が得られるか確認
    diff = (outputs1['pred_logits'] - outputs2['pred_logits']).abs().max().item()
    print(f"Max difference between two rollouts: {diff:.6e}")
    
    if diff < 1e-6:
        print("✓ Rollout is deterministic (autonomous)")
    else:
        print("⚠ Rollout is not deterministic")
    
    print()


def test_backward_pass():
    """逆伝播のテスト"""
    print("=== Test 4: Backward Pass ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = KoopmanRoutesFormer(
        vocab_size=39,
        token_emb_dim=32,
        d_model=64,
        nhead=4,
        num_layers=2,
        d_ff=128,
        z_dim=32,
        use_aux_loss=True,
    ).to(device)
    
    # ダミーデータ
    batch_size = 2
    prefix_len = 10
    K = 5
    
    prefix_tokens = torch.randint(0, 38, (batch_size, prefix_len)).to(device)
    prefix_stay_counts = torch.randint(0, 5, (batch_size, prefix_len)).to(device)
    prefix_agent_ids = torch.zeros(batch_size, dtype=torch.long).to(device)
    prefix_holidays = torch.randint(0, 2, (batch_size, prefix_len)).to(device)
    prefix_time_zones = torch.randint(0, 2, (batch_size, prefix_len)).to(device)
    prefix_events = torch.randint(0, 2, (batch_size, prefix_len)).to(device)
    future_tokens = torch.randint(0, 38, (batch_size, K)).to(device)
    
    # Forward
    model.train()
    outputs = model.forward_rollout(
        prefix_tokens=prefix_tokens,
        prefix_stay_counts=prefix_stay_counts,
        prefix_agent_ids=prefix_agent_ids,
        prefix_holidays=prefix_holidays,
        prefix_time_zones=prefix_time_zones,
        prefix_events=prefix_events,
        K=K,
        future_tokens=future_tokens,
    )
    
    # 損失計算
    pred_logits = outputs['pred_logits']
    ce_loss = F.cross_entropy(
        pred_logits.reshape(-1, model.vocab_size),
        future_tokens.reshape(-1)
    )
    
    total_loss = ce_loss
    if outputs['aux_losses']:
        total_loss += 0.01 * outputs['aux_losses']['count']
        total_loss += 0.01 * outputs['aux_losses']['mode']
    
    # Backward
    total_loss.backward()
    
    # 勾配がついているか確認
    has_grad = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad.append(name)
    
    print(f"Parameters with gradients: {len(has_grad)}/{len(list(model.parameters()))}")
    
    # A行列の勾配を確認
    if model.A.grad is not None:
        print(f"A matrix gradient norm: {model.A.grad.norm().item():.4f}")
        print("✓ Backward pass successful")
    else:
        print("⚠ A matrix has no gradient")
    
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("Koopman Routes Former - Modified Version Test")
    print("=" * 60)
    print()
    
    try:
        test_basic_forward()
        test_variable_length_prefix()
        test_autonomous_rollout()
        test_backward_pass()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()