import torch
import sacrebleu
from tqdm import tqdm
from janome.tokenizer import Tokenizer

# --- IMPORT TỪ CÁC FILE CỦA BẠN ---
from model import Transformer
from dataset import CustomTranslationDataset
# Import lại hàm beam_search từ file predict.py lúc nãy
from predict import beam_search_decode 

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & THÔNG SỐ
# ==========================================
TEST_SRC_FILE = "data/test.ja"
TEST_TGT_FILE = "data/test.vi"
OUTPUT_FILE = "predictions.txt" # File xuất kết quả dịch
CHECKPOINT_PATH = "checkpoints/best_model.pt"

# Các thông số phải khớp tuyệt đối với lúc Train
MAX_LEN = 50
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 1024
DROPOUT = 0.1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- ĐANG CHẠY ĐÁNH GIÁ TRÊN: {device} ---")

    # ==========================================
    # 2. KHÔI PHỤC TỪ ĐIỂN VÀ MÔ HÌNH
    # ==========================================
    print("1. Đang nạp bộ từ vựng (Vocabulary)...")
    # QUAN TRỌNG: Bạn bắt buộc phải lấy từ điển từ tập Train, 
    # KHÔNG được tạo từ điển mới từ tập Test để tránh sai lệch ID.
    train_dataset = CustomTranslationDataset("data/train.ja", "data/train.vi")
    src_vocab = train_dataset.src_vocab
    tgt_vocab = train_dataset.tgt_vocab
    tokenizer_ja = Tokenizer()

    print("2. Đang khởi tạo kiến trúc và nạp trọng số...")
    model = Transformer(
        src_vocab_size=src_vocab.vocab_size,
        tgt_vocab_size=tgt_vocab.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_length=MAX_LEN,
        dropout=DROPOUT
    ).to(device)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Chuyển sang chế độ suy luận
    print("✅ Đã nạp Model thành công!\n")

    # ==========================================
    # 3. ĐỌC DỮ LIỆU TEST
    # ==========================================
    with open(TEST_SRC_FILE, 'r', encoding='utf-8') as f:
        src_lines = f.readlines()
        
    with open(TEST_TGT_FILE, 'r', encoding='utf-8') as f:
        tgt_lines = f.readlines()
        
    assert len(src_lines) == len(tgt_lines), "Lỗi: Số dòng của file test.ja và test.vi không bằng nhau!"

    # ==========================================
    # 4. CHẠY VÒNG LẶP DỊCH (INFERENCE)
    # ==========================================
    predictions = []
    references = [tgt.strip().replace("_", " ") for tgt in tgt_lines] # SacreBLEU so sánh trên chữ tự nhiên
    
    print(f"Bắt đầu dịch {len(src_lines)} câu trong tập Test:")
    
    # Dùng tqdm để bọc danh sách, tạo thanh tiến trình đẹp mắt
    for line in tqdm(src_lines, desc="Translating", unit="câu"):
        # Tách từ tiếng Nhật
        words = [token.surface for token in tokenizer_ja.tokenize(line.strip())]
        
        # Chuyển thành Tensor
        src_indices = src_vocab.encode(words)
        src_indices.append(src_vocab.word2idx["<EOS>"])
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
        
        # Dịch bằng Beam Search
        tgt_indices = beam_search_decode(model, src_tensor, beam_width=5)
        
        # Giải mã về tiếng Việt
        clean_indices = [idx for idx in tgt_indices if idx not in [0, 1, 2]]
        translated_words = tgt_vocab.decode(clean_indices)
        
        # Nối lại và bỏ dấu gạch dưới của PyVi (VD: "sinh_viên" -> "sinh viên")
        translated_text = " ".join(translated_words).replace("_", " ")
        predictions.append(translated_text)

    # ==========================================
    # 5. XUẤT FILE VÀ TÍNH ĐIỂM
    # ==========================================
    # Ghi kết quả ra file txt
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + "\n")
            
    print(f"\n✅ Đã lưu toàn bộ kết quả dịch vào file: {OUTPUT_FILE}")
    
    # Tính điểm BLEU
    # SacreBLEU yêu cầu references phải là list of lists: [[ref1, ref2, ...]]
    refs_for_bleu = [references] 
    bleu_result = sacrebleu.corpus_bleu(predictions, refs_for_bleu)
    
    print("="*40)
    print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (EVALUATION)")
    print("="*40)
    print(f"Tổng số câu test: {len(predictions)}")
    print(f"Điểm BLEU Score : {bleu_result.score:.2f}")
    print("="*40)

if __name__ == "__main__":
    main()