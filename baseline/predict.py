import torch
import torch.nn.functional as F
from janome.tokenizer import Tokenizer

# Import từ các file của bạn
from model import Transformer
from dataset import CustomTranslationDataset

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ (Phải Y HỆT lúc Train)
# ==========================================
MAX_LEN = 50
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 1024
DROPOUT = 0.1

CHECKPOINT_PATH = "checkpoints/best_model.pt"

# ==========================================
# 2. HÀM BEAM SEARCH (Đã tối ưu cho Custom Vocab)
# ==========================================
def beam_search_decode(model, src_tensor, beam_width=5, alpha=0.7, bos_idx=1, eos_idx=2):
    model.eval()
    device = src_tensor.device
    
    src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)
    with torch.no_grad():
        enc_output = model.dropout(model.positional_encoding(model.encoder_embedding(src_tensor)))
        for enc_layer in model.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            
    beams = [(torch.tensor([[bos_idx]], device=device), 0.0)]
    completed_beams = []
    
    for step in range(MAX_LEN):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_idx:
                completed_beams.append((seq, score))
                continue
                
            seq_len = seq.size(1)
            tgt_mask = (seq != 0).unsqueeze(1).unsqueeze(2)
            nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool().to(device)
            tgt_mask = tgt_mask & nopeak_mask
            
            with torch.no_grad():
                dec_output = model.dropout(model.positional_encoding(model.decoder_embedding(seq)))
                for dec_layer in model.decoder_layers:
                    dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
                logits = model.fc_out(dec_output[:, -1, :])
                
            log_probs = F.log_softmax(logits, dim=-1)
            topk_log_probs, topk_idx = torch.topk(log_probs, beam_width, dim=-1)
            
            for i in range(beam_width):
                next_word = topk_idx[0, i].unsqueeze(0).unsqueeze(0)
                new_score = score + topk_log_probs[0, i].item()
                new_seq = torch.cat([seq, next_word], dim=1)
                new_beams.append((new_seq, new_score))
                
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if len(completed_beams) >= beam_width:
            break
            
    if len(completed_beams) == 0:
        completed_beams = beams

    # Áp dụng Length Penalty
    best_seq = None
    best_final_score = float('-inf')
    for seq, score in completed_beams:
        length = seq.size(1) - 1 
        penalty = (length ** alpha) if length > 0 else 1.0
        penalized_score = score / penalty
        if penalized_score > best_final_score:
            best_final_score = penalized_score
            best_seq = seq
            
    return best_seq[0].tolist()

# ==========================================
# 3. HÀM DỊCH MỘT CÂU (Pipeline hoàn chỉnh)
# ==========================================
def translate_sentence(sentence, model, src_vocab, tgt_vocab, tokenizer_ja, device):
    # Bước 1: Tiền xử lý (Tách từ Janome)
    words = [token.surface for token in tokenizer_ja.tokenize(sentence.strip())]
    
    # Bước 2: Chuyển chữ thành số (ID) và thêm <EOS>
    src_indices = src_vocab.encode(words)
    src_indices.append(src_vocab.word2idx["<EOS>"])
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device) # Thêm chiều batch_size = 1
    
    # Bước 3: Chạy Beam Search lấy list ID đầu ra
    tgt_indices = beam_search_decode(model, src_tensor)
    
    # Bước 4: Giải mã (Số -> Chữ)
    clean_indices = [idx for idx in tgt_indices if idx not in [0, 1, 2]] # Bỏ PAD, BOS, EOS
    translated_words = tgt_vocab.decode(clean_indices)
    
    # Bước 5: Hậu xử lý (Nối từ và bỏ dấu gạch dưới của PyVi)
    translated_text = " ".join(translated_words).replace("_", " ")
    return translated_text
