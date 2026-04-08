from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from datasets import Dataset as dst
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def beam_search_decode(model, beam_size, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_initial_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # Create a candidate list
    candidates = [(decoder_initial_input, 1)]

    while True:

        # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand, _ in candidates]):
            break

        # Create a new list of candidates
        new_candidates = []

        for candidate, score in candidates:

            # Do not expand candidates that have reached the eos token
            if candidate[0][-1].item() == eos_idx:
                continue

            # Build the candidate's mask
            candidate_mask = causal_mask(candidate.size(1)).type_as(source_mask).to(device)
            # calculate output
            out = model.decode(encoder_output, source_mask, candidate, candidate_mask)
            # get next token probabilities
            prob = model.project(out[:, -1])
            # get the top k candidates
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
            for i in range(beam_size):
                # for each of the top k candidates, get the token and its probability
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                # create a new candidate by appending the token to the current candidate
                new_candidate = torch.cat([candidate, token], dim=1)
                # We sum the log probabilities because the probabilities are in log space
                new_candidates.append((new_candidate, score + token_prob))

        # Sort the new candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # Keep only the top k candidates
        candidates = candidates[:beam_size]

        # If all the candidates have reached the eos token, stop
        if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
            break

    # Return the best candidate
    return candidates[0][0].squeeze()

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            if (len(predicted) > 99):
                break

            if count <= num_examples:
                print_msg('-'*console_width)
                # Print the source, target and model output
                print_msg(f"{f'SOURCE: ':>12}{source_text}")
                print_msg(f"{f'TARGET: ':>12}{target_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
                
    print_msg('-'*console_width)
    # Compute the BLEU metric
    # Khởi tạo bộ chấm điểm BLEU-1
    metric_b1 = torchmetrics.BLEUScore(n_gram=1, weights=[1.0])

    # Khởi tạo bộ chấm điểm BLEU-2
    metric_b2 = torchmetrics.BLEUScore(n_gram=2, weights=[0.5, 0.5])
    metric_b3 = torchmetrics.BLEUScore(n_gram=3, weights=[1/3, 1/3, 1/3])
    metric_b4 = torchmetrics.BLEUScore(n_gram=4, weights=[0.25, 0.25, 0.25, 0.25])

    bleu1 = metric_b1(predicted, expected)
    bleu2 = metric_b2(predicted, expected)
    bleu3 = metric_b3(predicted, expected)
    bleu4 = metric_b4(predicted, expected)

    print(f"BLEU-1: {bleu1.item() * 100:.2f}")
    print(f"BLEU-2: {bleu2.item() * 100:.2f}")
    print(f"BLEU-3: {bleu3.item() * 100:.2f}")
    print(f"BLEU-4: {bleu4.item() * 100:.2f}")

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def run_test(model, test_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=5):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in test_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for test"

            model_out = beam_search_decode(model, 3, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            if (len(predicted) > 1000):
                break

            if count <= num_examples:
                print_msg('-'*console_width)
                # Print the source, target and model output
                print_msg(f"{f'SOURCE: ':>12}{source_text}")
                print_msg(f"{f'TARGET: ':>12}{target_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
                
    print_msg('-'*console_width)
    # Compute the BLEU metric
    metric_b1 = torchmetrics.BLEUScore(n_gram=1, weights=[1.0])

    # Khởi tạo bộ chấm điểm BLEU-2
    metric_b2 = torchmetrics.BLEUScore(n_gram=2, weights=[0.5, 0.5])
    metric_b3 = torchmetrics.BLEUScore(n_gram=3, weights=[1/3, 1/3, 1/3])
    metric_b4 = torchmetrics.BLEUScore(n_gram=4, weights=[0.25, 0.25, 0.25, 0.25])

    bleu1 = metric_b1(predicted, expected)
    bleu2 = metric_b2(predicted, expected)
    bleu3 = metric_b3(predicted, expected)
    bleu4 = metric_b4(predicted, expected)

    print(f"BLEU-1: {bleu1.item() * 100:.2f}")
    print(f"BLEU-2: {bleu2.item() * 100:.2f}")
    print(f"BLEU-3: {bleu3.item() * 100:.2f}")
    print(f"BLEU-4: {bleu4.item() * 100:.2f}")
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('test cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('test wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('test BLEU', bleu, global_step)
        writer.flush()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def ds_custom():
    # 1. Khai báo đường dẫn tới 2 file text của bạn
    # (Hãy thay đổi tên file và mã ngôn ngữ cho đúng với bài toán của bạn)
    file_src_path = 'dataset/train.ja-vi.ja' 
    file_tgt_path = 'dataset/train.ja-vi.vi'

    lang_src = 'ja'
    lang_tgt = 'vi'

    # 2. Đọc dữ liệu từ 2 file
    with open(file_src_path, 'r', encoding='utf-8') as f_src, \
        open(file_tgt_path, 'r', encoding='utf-8') as f_tgt:
        
        # Đọc toàn bộ các dòng vào bộ nhớ
        lines_src = f_src.readlines()
        lines_tgt = f_tgt.readlines()

    # 3. Kiểm tra an toàn (Bắt buộc)
    # Trong dịch máy, nếu 2 file lệch nhau 1 dòng thì toàn bộ dữ liệu phía sau sẽ bị ghép sai cặp.
    assert len(lines_src) == len(lines_tgt), f"Lỗi: File nguồn có {len(lines_src)} dòng, nhưng file đích có {len(lines_tgt)} dòng!"

    # 4. Ghép cặp và tạo cấu trúc từ điển lồng nhau
    formatted_data = []
    for src_text, tgt_text in zip(lines_src, lines_tgt):
        
        # .strip() cực kỳ quan trọng để xóa ký tự xuống dòng (\n) ẩn ở cuối mỗi câu trong file text
        clean_src = src_text.strip()
        clean_tgt = tgt_text.strip()
        
        # Chỉ thêm vào dataset nếu cả 2 câu đều không bị trống
        if clean_src and clean_tgt:
            formatted_data.append({
                "translation": {
                    lang_src: clean_src,
                    lang_tgt: clean_tgt
                }
            })

    # 5. Phép thuật của Hugging Face: Biến List thành Dataset
    ds_raw = dst.from_list(formatted_data)

    # Kiểm tra thành quả
    # print(f"Đã tạo thành công Dataset với {len(ds_raw)} cặp câu!")
    # print("Mẫu dữ liệu dòng đầu tiên:", ds_raw[0])
    return ds_raw

def ds_custom_val():
    # 1. Khai báo đường dẫn tới 2 file text của bạn
    # (Hãy thay đổi tên file và mã ngôn ngữ cho đúng với bài toán của bạn)
    file_src_path = 'dataset/dev2010.ja-vi.ja' 
    file_tgt_path = 'dataset/dev2010.ja-vi.vi'

    lang_src = 'ja'
    lang_tgt = 'vi'

    # 2. Đọc dữ liệu từ 2 file
    with open(file_src_path, 'r', encoding='utf-8') as f_src, \
        open(file_tgt_path, 'r', encoding='utf-8') as f_tgt:
        
        # Đọc toàn bộ các dòng vào bộ nhớ
        lines_src = f_src.readlines()
        lines_tgt = f_tgt.readlines()

    # 3. Kiểm tra an toàn (Bắt buộc)
    # Trong dịch máy, nếu 2 file lệch nhau 1 dòng thì toàn bộ dữ liệu phía sau sẽ bị ghép sai cặp.
    assert len(lines_src) == len(lines_tgt), f"Lỗi: File nguồn có {len(lines_src)} dòng, nhưng file đích có {len(lines_tgt)} dòng!"

    # 4. Ghép cặp và tạo cấu trúc từ điển lồng nhau
    formatted_data = []
    for src_text, tgt_text in zip(lines_src, lines_tgt):
        
        # .strip() cực kỳ quan trọng để xóa ký tự xuống dòng (\n) ẩn ở cuối mỗi câu trong file text
        clean_src = src_text.strip()
        clean_tgt = tgt_text.strip()
        
        # Chỉ thêm vào dataset nếu cả 2 câu đều không bị trống
        if clean_src and clean_tgt:
            formatted_data.append({
                "translation": {
                    lang_src: clean_src,
                    lang_tgt: clean_tgt
                }
            })

    # 5. Phép thuật của Hugging Face: Biến List thành Dataset
    ds_raw = dst.from_list(formatted_data)

    # Kiểm tra thành quả
    # print(f"Đã tạo thành công Dataset với {len(ds_raw)} cặp câu!")
    # print("Mẫu dữ liệu dòng đầu tiên:", ds_raw[0])
    return ds_raw

def ds_custom_test():
    # 1. Khai báo đường dẫn tới 2 file text của bạn
    # (Hãy thay đổi tên file và mã ngôn ngữ cho đúng với bài toán của bạn)
    file_src_path = 'dataset/tst2010.ja-vi.ja' 
    file_tgt_path = 'dataset/tst2010.ja-vi.vi'

    lang_src = 'ja'
    lang_tgt = 'vi'

    # 2. Đọc dữ liệu từ 2 file
    with open(file_src_path, 'r', encoding='utf-8') as f_src, \
        open(file_tgt_path, 'r', encoding='utf-8') as f_tgt:
        
        # Đọc toàn bộ các dòng vào bộ nhớ
        lines_src = f_src.readlines()
        lines_tgt = f_tgt.readlines()

    # 3. Kiểm tra an toàn (Bắt buộc)
    # Trong dịch máy, nếu 2 file lệch nhau 1 dòng thì toàn bộ dữ liệu phía sau sẽ bị ghép sai cặp.
    assert len(lines_src) == len(lines_tgt), f"Lỗi: File nguồn có {len(lines_src)} dòng, nhưng file đích có {len(lines_tgt)} dòng!"

    # 4. Ghép cặp và tạo cấu trúc từ điển lồng nhau
    formatted_data = []
    for src_text, tgt_text in zip(lines_src, lines_tgt):
        
        # .strip() cực kỳ quan trọng để xóa ký tự xuống dòng (\n) ẩn ở cuối mỗi câu trong file text
        clean_src = src_text.strip()
        clean_tgt = tgt_text.strip()
        
        # Chỉ thêm vào dataset nếu cả 2 câu đều không bị trống
        if clean_src and clean_tgt:
            formatted_data.append({
                "translation": {
                    lang_src: clean_src,
                    lang_tgt: clean_tgt
                }
            })

    # 5. Phép thuật của Hugging Face: Biến List thành Dataset
    ds_raw = dst.from_list(formatted_data)

    # Kiểm tra thành quả
    # print(f"Đã tạo thành công Dataset với {len(ds_raw)} cặp câu!")
    # print("Mẫu dữ liệu dòng đầu tiên:", ds_raw[0])
    return ds_raw

def get_ds(config):
    # It only has the train split, so we divide it overselves
    # ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    ds_raw = ds_custom()
    ds_raw_val = ds_custom_val()
    ds_raw_test = ds_custom_test()
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = len(ds_raw)
    val_ds_size = len(ds_raw_val)
    test_ds_size = len(ds_raw_test)
    print(f"The dataset has {train_ds_size} sentences in train data.")
    print(f"The dataset has {val_ds_size} sentences in valid data.")
    print(f"The dataset has {test_ds_size} sentences in test data.")


    train_ds_raw, val_ds_raw, test_ds_raw = ds_raw, ds_raw_val, ds_raw_test

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_ds = BilingualDataset(test_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)


    return train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader,test_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tổng số tham số có thể huấn luyện (Trainable Parameters)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('-'*50)
    print(f"The number parameters of model: {total_params:,}")
    print('-'*50)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    # Khởi tạo kỷ lục Val Loss ban đầu là Vô cực
    best_val_loss = float('inf')
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
        # ==========================================
        # PHẦN 2: ĐÁNH GIÁ (VALIDATION)
        # ==========================================
        # Đảm bảo bạn đã khai báo val_dataloader ở trên cùng với train_dataloader
        
        model.eval() # TẮT Dropout và Batch Norm
        val_loss_total = 0.0
        
        # TẮT tính toán đạo hàm (Gradient) để tiết kiệm 50% RAM và tăng tốc
        with torch.no_grad():
            val_iterator = tqdm(val_dataloader, desc=f"Validating Epoch {epoch:02d}", colour="green")
            
            for batch in val_iterator:
                # Đẩy dữ liệu lên GPU
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)
                label = batch['label'].to(device)

                # Lan truyền xuôi (Forward pass) y hệt như Train
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)

                # Tính Loss
                val_loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                val_loss_total += val_loss.item()
                
        # Tính trung bình Validation Loss của toàn bộ Epoch
        avg_val_loss = val_loss_total / len(val_dataloader)
        
        # Ghi log lên TensorBoard 
        # (Lưu ý: Dùng cùng global_step để 2 đường đồ thị khớp với nhau trên trục X)
        writer.add_scalar('val_loss', avg_val_loss, global_step)
        writer.flush()
        
        # In tổng kết Epoch ra màn hình
        print(f"✅ Epoch {epoch:02d} Completed | Avg Val Loss: {avg_val_loss:.4f}")

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        if epoch == config['num_epochs'] - 1:
            print("---Evaluate Model With Test Dataset----")
            run_test(model, test_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        # Save the model at the end of every epoch
        if avg_val_loss < best_val_loss:
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)
    
    


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
