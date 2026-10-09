# dataset.py (обновленная версия create_dataset)
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizator import tokenizer_gpt2

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        
        # Метод скользящего окна
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataset(train_filename: str, valid_filename: str, model_cfg: dict, train_ratio=0.90):
    print(f"Чтение данных из {train_filename}...")
    with open(train_filename, "r", encoding="utf-8") as file:
        text_data_train = file.read()

    with open(valid_filename, "r", encoding="utf-8") as file:
        text_data_valid = file.read()

    print(f"Размер текста для обучения: {len(text_data_train)} символов.")
    print(f"Размер текста для тестирования: {len(text_data_valid)} символов.")
    
    split_idx = int(train_ratio * len(text_data_train))
    train_data = text_data_train[:split_idx]
    val_data = text_data_train[split_idx:]

    print("Создание тренировочного DataLoader...")
    train_loader = DataLoader(
        GPTDatasetV1(train_data, tokenizer_gpt2, model_cfg["context_length"], model_cfg["context_length"]),
        batch_size=4, # Можно увеличить до 8 или 16, если позволяет VRAM
        shuffle=True,
        drop_last=True,
        num_workers=0 # На Windows лучше 0, на Linux можно поставить 2-4
    )
    
    print("Создание валидационного DataLoader...")
    val_loader = DataLoader(
        GPTDatasetV1(val_data, tokenizer_gpt2, model_cfg["context_length"], model_cfg["context_length"]),
        batch_size=4,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    return train_loader, val_loader