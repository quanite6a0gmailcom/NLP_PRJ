import fugashi
from pyvi import ViTokenizer
from sacremoses import MosesPunctNormalizer
from tqdm import tqdm

tagger = fugashi.Tagger()
normalizer = MosesPunctNormalizer(lang='en')

def process_japanese_line(text):
    words = [word.surface for word in tagger(text.strip())]
    return " ".join(words)

def process_vietnamese_line(text):
    # Bước 1: Dọn rác
    clean_text = normalizer.normalize(text.strip())
    # Bước 2: Cắt từ (Đã xóa dòng gán đè sai logic)
    # final_text = ViTokenizer.tokenize(clean_text)
    return final_text

input_ja_file = "dataset/all.ja"
input_vi_file = "dataset/all.vi"
output_ja_file = "dataset_bpemb/all.ja"
output_vi_file = "dataset_bpemb/all.vi"

print("Bắt đầu xử lý đồng bộ Dataset...")

# Mở cả 4 file cùng một lúc
with open(input_ja_file, 'r', encoding='utf-8') as f_ja_in, \
     open(input_vi_file, 'r', encoding='utf-8') as f_vi_in, \
     open(output_ja_file, 'w', encoding='utf-8') as f_ja_out, \
     open(output_vi_file, 'w', encoding='utf-8') as f_vi_out:
    
    # Dùng zip() để đọc từng cặp dòng (1 Nhật - 1 Việt) cùng lúc
    for line_ja, line_vi in tqdm(zip(f_ja_in, f_vi_in)):
        
        # KIỂM TRA ĐỒNG BỘ: 
        # Nếu MỘT TRONG HAI câu bị trống (do lỗi dataset), ta vứt bỏ cả cặp
        if not line_ja.strip() or not line_vi.strip():
            continue 
            
        # Xử lý dữ liệu
        processed_ja = process_japanese_line(line_ja)
        processed_vi = process_vietnamese_line(line_vi)
        
        # BƯỚC CHỐNG LỆCH DÒNG CUỐI CÙNG: 
        # Đề phòng trường hợp sau khi xử lý (vd: xóa hết icon), câu bỗng nhiên trống trơn
        if not processed_ja.strip() or not processed_vi.strip():
            continue
            
        # Ghi vào file output cùng một lúc
        f_ja_out.write(processed_ja + "\n")
        f_vi_out.write(processed_vi + "\n")

print("✅ Đã xử lý xong! Hai file đảm bảo khớp nhau 100%.")