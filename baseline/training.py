import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformer_layers import *

# 1. Định nghĩa các Special Tokens
PAD_IDX = 0  # Padding
BOS_IDX = 1  # Beginning of Sequence
EOS_IDX = 2  # End of Sequence

# Giả lập tham số từ vựng (ví dụ cho bài toán dịch máy)
src_vocab_size = 5000
tgt_vocab_size = 7000
max_seq_length = 50
batch_size = 32

# 2. Tạo Dataset giả lập (Dummy Dataset)
class DummyTranslationDataset(Dataset):
    def __init__(self, num_samples, max_len):
        self.num_samples = num_samples
        # Sinh ngẫu nhiên dữ liệu, đảm bảo có chứa PAD ở cuối
        self.src_data = torch.randint(3, src_vocab_size, (num_samples, max_len))
        self.tgt_data = torch.randint(3, tgt_vocab_size, (num_samples, max_len))
        
        # Thêm BOS và EOS vào target để huấn luyện
        self.tgt_data[:, 0] = BOS_IDX
        self.tgt_data[:, -1] = EOS_IDX

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

dataset = DummyTranslationDataset(num_samples=1000, max_len=max_seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Cấu hình thiết bị (Sử dụng GPU/CUDA nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình Transformer (sử dụng class đã định nghĩa ở phần trước)
model = TransformerModel(
    src_vocab_size=src_vocab_size, 
    tgt_vocab_size=tgt_vocab_size, 
    d_model=512, 
    num_heads=8, 
    num_layers=6, 
    d_ff=2048, 
    max_seq_length=max_seq_length, 
    dropout=0.1
).to(device)

# 3. Khởi tạo Optimizer và Loss Function
# Sử dụng Adam optimizer với tham số chuẩn từ bài báo
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Quan trọng: ignore_index=PAD_IDX giúp model không tính loss cho các token Padding
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# 4. Bắt đầu vòng lặp huấn luyện
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train() # Chuyển mô hình sang chế độ huấn luyện (kích hoạt Dropout, LayerNorm)
    epoch_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Kỹ thuật Teacher Forcing
        # Input cho decoder: từ đầu đến gần cuối (không có EOS)
        tgt_input = tgt[:, :-1]
        
        # Target kỳ vọng để tính loss: từ vị trí thứ 2 đến cuối (không có BOS)
        tgt_expected = tgt[:, 1:]
        
        # Xóa dốc gradient của batch trước
        optimizer.zero_grad()
        
        # Forward pass (Truyền xuôi)
        # Kích thước output: (batch_size, seq_length - 1, tgt_vocab_size)
        output = model(src, tgt_input)
        
        # Reshape lại để đưa vào hàm CrossEntropyLoss
        # Output dẹt: (batch_size * (seq_length - 1), tgt_vocab_size)
        # Expected dẹt: (batch_size * (seq_length - 1))
        output_flatten = output.contiguous().view(-1, tgt_vocab_size)
        tgt_expected_flatten = tgt_expected.contiguous().view(-1)
        
        # Tính toán độ lỗi (Loss)
        loss = criterion(output_flatten, tgt_expected_flatten)
        
        # Backward pass (Truyền ngược) và cập nhật trọng số
        loss.backward()
        
        # Có thể thêm Gradient Clipping ở đây để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch+1}/{EPOCHS} | Batch: {batch_idx} | Loss: {loss.item():.4f}")
            
    print(f"==> Kết thúc Epoch {epoch+1} | Loss trung bình: {epoch_loss / len(dataloader):.4f}\n")