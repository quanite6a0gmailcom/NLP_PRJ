from pyvi import ViTokenizer
import torch
from torch.utils.data import Dataset
from janome.tokenizer import Tokenizer
from vocabulary import *
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class CustomTranslationDataset(Dataset):
    def __init__(self,src_file,tgt_file):
        # Create Janome for Japanese
        self.tokenizer_ja = Tokenizer()

        self.src_data = []
        self.tgt_data = []

        print("---Processing Japanese Data---")
        # Read and Tokenize Japanese with Janome
        with open(src_file, 'r', encoding= 'utf-8') as f:
            for line in f:
                words = [token.surface for token in self.tokenizer_ja.tokenize(line.strip())] 
                self.src_data.append(words)
        
        print("---Processing Vietnamese Data---")
        #Read and Tokenize Vietnamese with Pyvi
        with open(tgt_file, 'r', encoding= 'utf-8') as f:
            for line in f:
                tokenized_line = ViTokenizer.tokenize(line.strip())
                self.tgt_data.append(tokenized_line.split())

        assert len(self.src_data) == len(self.tgt_data), "The number of lines in two files is inconsistent!"

        print("---Building vocabulary---")
        self.src_vocab = Vocabulary()
        self.tgt_vocab = Vocabulary()

        self.src_vocab.build_vocab(self.src_data)
        self.tgt_vocab.build_vocab(self.tgt_data)
        print("---Completed prepare data---")
    
    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_encoded = self.src_vocab.encode(self.src_data[idx])
        tgt_encoded = self.tgt_vocab.encode(self.tgt_data[idx])

        src_tensor = torch.tensor(src_encoded + [self.src_vocab.word2idx["<EOS>"]], dtype=torch.long)
        tgt_tensor = torch.tensor([self.tgt_vocab.word2idx["<BOS>"]] + tgt_encoded + [self.tgt_vocab.word2idx["<EOS>"]], dtype=torch.long)

        return src_tensor, tgt_tensor
    
import torch

def collate_fn(batch):
    PAD_IDX = 0
    EOS_IDX = 2 # ID của token <EOS> theo từ điển của bạn
    MAX_LEN = 50 # Chốt cứng độ dài tối đa cho mỗi câu
    
    batch_size = len(batch)
    
    # 1. Khởi tạo "khuôn" ma trận vuông vức chứa toàn số 0 (PAD)
    # Kích thước luôn luôn là (batch_size, MAX_LEN)
    src_padded = torch.full((batch_size, MAX_LEN), PAD_IDX, dtype=torch.long)
    tgt_padded = torch.full((batch_size, MAX_LEN), PAD_IDX, dtype=torch.long)
    
    for i, (src_item, tgt_item) in enumerate(batch):
        # --- XỬ LÝ SOURCE ---
        # Lấy chiều dài thực tế, nhưng không được vượt quá MAX_LEN
        src_len = min(len(src_item), MAX_LEN)
        
        # Đổ dữ liệu vào khuôn
        src_padded[i, :src_len] = src_item[:src_len]
        
        # CỨU HỘ QUAN TRỌNG: Nếu câu bị cắt cụt, từ cuối cùng bị mất <EOS>.
        # Ta phải gán lại <EOS> vào vị trí cuối để mô hình biết điểm dừng.
        if len(src_item) > MAX_LEN:
            src_padded[i, -1] = EOS_IDX

        # --- XỬ LÝ TARGET ---
        tgt_len = min(len(tgt_item), MAX_LEN)
        tgt_padded[i, :tgt_len] = tgt_item[:tgt_len]
        
        if len(tgt_item) > MAX_LEN:
            tgt_padded[i, -1] = EOS_IDX
            
    return src_padded, tgt_padded



# 2. Khởi tạo Dataset
print("--- ĐANG KHỞI TẠO DATASET ---")
dataset = CustomTranslationDataset("C:\\NLP_PRJ\\TEDjavi_106K\\dev2010.ja-vi.ja", "C:\\NLP_PRJ\\TEDjavi_106K\\dev2010.ja-vi.vi")

# 3. Kiểm tra kích thước Vocab
print("\n[1] KÍCH THƯỚC TỪ VỰNG")
print(f"- Vocab Tiếng Nhật: {dataset.src_vocab.vocab_size} từ")
print(f"- Vocab Tiếng Việt: {dataset.tgt_vocab.vocab_size} từ")

# 4. In thử một vài từ trong từ điển (Word -> ID)
print("\n[2] BẢNG ÁNH XẠ TỪ ĐIỂN (10 từ đầu tiên)")
# Ép kiểu list để chỉ lấy 10 phần tử đầu tiên in ra màn hình
print("- Tiếng Nhật:", list(dataset.src_vocab.word2idx.items())[:10])
print("- Tiếng Việt:", list(dataset.tgt_vocab.word2idx.items())[:10])

# 5. Test tính năng Encode (Mã hóa) và Decode (Giải mã)
print("\n[3] KIỂM TRA QUY TRÌNH MÃ HÓA & GIẢI MÃ")

# Test với Tiếng Nhật
sample_ja = ["私", "は", "学生", "です", "。"]
encoded_ja = dataset.src_vocab.encode(sample_ja)
decoded_ja = dataset.src_vocab.decode(encoded_ja)
print(f"JA Gốc     : {sample_ja}")
print(f"JA Mã hóa  : {encoded_ja}")
print(f"JA Giải mã : {decoded_ja}")

# Test với Tiếng Việt
sample_vi = ["Tôi", "là", "sinh_viên", "."]
encoded_vi = dataset.tgt_vocab.encode(sample_vi)
decoded_vi = dataset.tgt_vocab.decode(encoded_vi)
print(f"\nVI Gốc     : {sample_vi}")
print(f"VI Mã hóa  : {encoded_vi}")
print(f"VI Giải mã : {decoded_vi}")

# 6. Test từ OOV (Out-Of-Vocabulary) để xem có ra ID của <UNK> (số 3) không
print("\n[4] KIỂM TRA TỪ CHƯA TỪNG XUẤT HIỆN (OOV)")
oov_word = ["Doraemon"]
print(f"Từ lạ '{oov_word}' bị mã hóa thành ID: {dataset.src_vocab.encode(oov_word)}")

# 7. Xem thử Tensor đầu ra của Dataset (Đã gắn BOS và EOS chưa)
print("\n[5] KIỂM TRA TENSOR ĐẦU RA")
src_tensor, tgt_tensor = dataset[0] # Lấy cặp câu đầu tiên
print(f"- Source Tensor (kết thúc bằng EOS {dataset.src_vocab.word2idx['<EOS>']}):\n  {src_tensor}")
print(f"- Target Tensor (bắt đầu bằng BOS {dataset.tgt_vocab.word2idx['<BOS>']}, kết thúc bằng EOS):\n  {tgt_tensor}")

print("\n[6] KIỂM TRA QUÁ TRÌNH CHIA BATCH (DATALOADER)")

# Cài đặt kích thước Batch (Ví dụ: 2 câu một lượt)
BATCH_SIZE = 2

# Khởi tạo DataLoader
train_dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,           # Trộn ngẫu nhiên dữ liệu ở mỗi Epoch
    collate_fn=collate_fn,  # Gọi hàm đệm động
    drop_last=False         # Nếu batch cuối không đủ 2 câu thì vẫn lấy
)

# Chạy thử 1 vòng lặp để kéo Batch đầu tiên ra kiểm tra
for batch_idx, (src_batch, tgt_batch) in enumerate(train_dataloader):
    print(f"--- BATCH {batch_idx + 1} ---")
    print(f"Kích thước Source Tensor: {src_batch.shape}")
    print(f"Kích thước Target Tensor: {tgt_batch.shape}")
    
    print("\nChi tiết Source Tensor (Lưu ý các số 0 ở cuối nếu bị đệm):")
    print(src_batch)
    
    print("\nChi tiết Target Tensor:")
    print(tgt_batch)
    
    # Chỉ in 1 batch rồi dừng để kiểm tra
    break