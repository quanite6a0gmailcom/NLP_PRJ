import math
from collections import Counter
import scipy.stats as stats

def calculate_ribes_sentence(pred, ref, alpha=0.25, beta=0.10):
    """
    Tính điểm RIBES cho một cặp câu.
    """
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    
    if not pred_tokens or not ref_tokens:
        return 0.0

    # 1. Tìm các từ khớp nhau (Unigram matching)
    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    
    matches = []
    # Lưu lại vị trí của các từ khớp nhau để tính Rank
    ref_word_to_idx = {word: idx for idx, word in enumerate(ref_tokens)}
    pred_aligned_indices = []
    
    for word in pred_tokens:
        if ref_counts[word] > 0:
            matches.append(word)
            pred_aligned_indices.append(ref_word_to_idx[word])
            ref_counts[word] -= 1
            
    matched_count = len(matches)
    if matched_count == 0:
        return 0.0

    # 2. Tính Unigram Precision (P) và Recall (R)
    precision = matched_count / len(pred_tokens)
    recall = matched_count / len(ref_tokens)
    
    # Tính F-score (chú trọng Precision hơn thông qua beta)
    f_score = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)
    
    # 3. Tính độ tương quan hạng Kendall's Tau
    # So sánh trật tự từ trong Pred so với trật tự chuẩn trong Ref
    ideal_order = list(range(len(pred_aligned_indices)))
    
    # Nếu chỉ có 1 từ khớp, tau = 1.0 (tránh lỗi chia cho 0)
    if len(pred_aligned_indices) > 1:
        tau, _ = stats.kendalltau(pred_aligned_indices, ideal_order)
        # Chuẩn hóa Tau từ [-1, 1] về [0, 1]
        n_tau = (tau + 1) / 2
    else:
        n_tau = 1.0
        
    # 4. Tính toán điểm RIBES cuối cùng
    ribes_score = n_tau * (precision ** alpha)
    return ribes_score

def corpus_ribes(preds, refs):
    """Tính điểm RIBES trung bình cho toàn bộ tập dữ liệu"""
    total_score = 0
    for p, r in zip(preds, refs[0]): # Giả sử có 1 list reference
        total_score += calculate_ribes_sentence(p, r)
        
    return (total_score / len(preds)) * 100 # Chuyển về thang 100