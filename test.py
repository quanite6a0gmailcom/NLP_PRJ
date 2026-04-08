import os
import time
from bpemb import BPEmb

# Nhớ khai báo lại đường dẫn của bạn ở đây
path_to_model_vi = "bpemb_vn/vi.wiki.bpe.vs25000.model"
path_to_txt_vi = "bpemb_vn/vi.wiki.bpe.vs25000.d300.w2v.txt"

print("--- KIỂM TRA ĐƯỜNG DẪN ---")
print(f"1. File model tồn tại: {os.path.exists(path_to_model_vi)}")
print(f"2. File txt tồn tại: {os.path.exists(path_to_txt_vi)}")

if not os.path.exists(path_to_model_vi) or not os.path.exists(path_to_txt_vi):
    print("❌ LỖI: Đường dẫn bị sai. Vui lòng kiểm tra lại cấu trúc thư mục!")
else:
    print("\nĐang load BPEmb vào RAM (Hãy kiên nhẫn chờ 15 - 40 giây)...")
    start_time = time.time()
    
    try:
        tokenizer_tgt = BPEmb(
            lang="vi",
            vs=25000,
            dim=300,
            model_file=path_to_model_vi,
            emb_file=path_to_txt_vi
        )
        end_time = time.time()
        print(f"✅ Tải thành công! Mất tổng cộng: {end_time - start_time:.2f} giây.")
        print(f"Kích thước vector: {tokenizer_tgt.vectors.shape}")
        
    except Exception as e:
        print(f"❌ Có lỗi xảy ra trong quá trình load: {e}")