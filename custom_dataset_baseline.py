import fugashi
from pyvi import ViTokenizer
from sacremoses import MosesPunctNormalizer
from tqdm import tqdm

# Khởi tạo công cụ
# tagger = fugashi.Tagger()
normalizer = MosesPunctNormalizer(lang='en')

# def process_japanese_line(text):
#     # Dùng fugashi cho tiếng Nhật
#     words = [word.surface for word in tagger(text.strip())]
#     return " ".join(words)

def process_vietnamese_line(text):
    # BƯỚC 1: Chuẩn hóa bằng Moses (Dọn rác Unicode, dấu câu)
    clean_text = normalizer.normalize(text.strip())
    
    # BƯỚC 2: Cắt từ bằng Pyvi (Tạo dấu gạch nối _)
    # final_text = ViTokenizer.tokenize(clean_text)
    final_text = clean_text
    
    return final_text

# 2. Cấu hình đường dẫn file
input_ja_file = "dataset/tst2010.ja-vi.ja"
input_vi_file = "dataset/tst2010.ja-vi.vi"

output_ja_file = "dataset_bpemb/tst2010.ja-vi.ja"
output_vi_file = "dataset_bpemb/tst2010.ja-vi.vi"

# # 3. Tiến hành đọc, xử lý và ghi ra file mới
# print("Bắt đầu xử lý tiếng Nhật...")
# with open(input_ja_file, 'r', encoding='utf-8') as f_in, \
#      open(output_ja_file, 'w', encoding='utf-8') as f_out:
    
#     # Đọc từng dòng, xử lý, rồi ghi luôn vào file mới để không bị tràn RAM
#     for line in tqdm(f_in):
#         if line.strip(): # Bỏ qua dòng trống
#             processed_line = process_japanese_line(line)
#             f_out.write(processed_line + "\n")

print("Bắt đầu xử lý tiếng Việt...")
with open(input_vi_file, 'r', encoding='utf-8') as f_in, \
     open(output_vi_file, 'w', encoding='utf-8') as f_out:
    
    for line in tqdm(f_in):
        if line.strip():
            processed_line = process_vietnamese_line(line)
            f_out.write(processed_line + "\n")

print("✅ Đã xử lý xong toàn bộ Dataset!")