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

