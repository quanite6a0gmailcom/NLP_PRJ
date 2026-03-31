import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformer_layers import *
from dataset import *
import os

SRC_FILE_TRAIN = "C:\\NLP_PRJ\\TEDjavi_106K\\dev2010.ja-vi.ja"
TGT_FILE_TRAIN = "C:\\NLP_PRJ\\TEDjavi_106K\\dev2010.ja-vi.vi"
SRC_FILE_VAL = "C:\\NLP_PRJ\\TEDjavi_106K\\dev2010.ja-vi.ja"
TGT_FILE_VAL = "C:\\NLP_PRJ\\TEDjavi_106K\\dev2010.ja-vi.vi"

BATCH_SIZE = 32
MAX_LEN = 50
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 1024
DROPOUT = 0.1
EPOCHS = 50
LEARNING_RATE = 0.001

def count_parameters(model):
    """Đếm tổng số tham số có thể cập nhật trong mô hình"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# ============================================
# 2. DEFINE TRAIN FUNCTION & EVALUATE FUNCTION
# ============================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]
        
        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        output_dim = output.shape[-1]
        output_flatten = output.contiguous().view(-1, output_dim)
        tgt_expected_flatten = tgt_expected.contiguous().view(-1)
        
        loss = criterion(output_flatten, tgt_expected_flatten)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"   [Train] Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
    return epoch_loss / len(dataloader)

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval() # Tắt Dropout và LayerNorm
    epoch_loss = 0
    with torch.no_grad(): # Không tính Gradient để tiết kiệm RAM
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]
            
            output = model(src, tgt_input)
            
            output_dim = output.shape[-1]
            output_flatten = output.contiguous().view(-1, output_dim)
            tgt_expected_flatten = tgt_expected.contiguous().view(-1)
            
            loss = criterion(output_flatten, tgt_expected_flatten)
            epoch_loss += loss.item()
            
    return epoch_loss / len(dataloader)

# ==========================================
# 3. VÒNG LẶP CHÍNH (MAIN LOOP)
# ==========================================
if __name__ == "__main__":
    # Thiết lập thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")
    
    # Tạo thư mục lưu model nếu chưa có
    os.makedirs("checkpoints", exist_ok=True)

    print("\n--- BƯỚC 1: CHUẨN BỊ DỮ LIỆU ---")
    train_dataset = CustomTranslationDataset(SRC_FILE_TRAIN, TGT_FILE_TRAIN)
    # Tái sử dụng Vocab của tập Train cho tập Val để tránh lệch ID
    val_dataset = CustomTranslationDataset(SRC_FILE_VAL, TGT_FILE_VAL)
    val_dataset.src_vocab = train_dataset.src_vocab
    val_dataset.tgt_vocab = train_dataset.tgt_vocab

    SRC_VOCAB_SIZE = train_dataset.src_vocab.vocab_size
    TGT_VOCAB_SIZE = train_dataset.tgt_vocab.vocab_size
    print(f"Vocab Source (Nhật): {SRC_VOCAB_SIZE} | Vocab Target (Việt): {TGT_VOCAB_SIZE}")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("\n--- BƯỚC 2: KHỞI TẠO MÔ HÌNH ---")
    model = TransformerModel(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_length=MAX_LEN,
        dropout=DROPOUT
    ).to(device)

    total_params = count_parameters(model)
    print(f"Tổng số tham số của mô hình: {total_params:,} parameters")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    # ignore_index=0 để bỏ qua padding
    criterion = nn.CrossEntropyLoss(ignore_index=0) 

    print("\n--- BƯỚC 3: BẮT ĐẦU HUẤN LUYỆN ---")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss = evaluate_epoch(model, val_dataloader, criterion, device)
        
        print(f"==> Tổng kết Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Lưu checkpoint nếu Validation Loss giảm (Mô hình tốt lên)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = f"checkpoints/best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"*** Đã lưu mô hình tốt nhất tại: {checkpoint_path} ***")

