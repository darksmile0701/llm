import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tokenizers.normalizers import NFKC

def train_custom_bpe(input_files, vocab_size=30000, output_path="russian_bpe_tokenizer.json"):
    print(f"Начало обучения токенизатора на файлах: {input_files}")
    
    # Инициализируем пустой BPE токенизатор
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # Нормализация (приводит разные виды кавычек и пробелов к единому стандарту)
    tokenizer.normalizer = NFKC()
    
    # Предварительное разбиение на слова (по пробелам и пунктуации)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    # Декодер для превращения токенов обратно в текст
    tokenizer.decoder = decoders.ByteLevel()
    
    # Настройки тренировщика
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2, # Игнорировать символы, встречающиеся реже 2 раз
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    # Запуск обучения
    tokenizer.train(input_files, trainer)
    
    # Сохранение
    tokenizer.save(output_path)
    print(f"Токенизатор успешно обучен и сохранен в {output_path}")
    print(f"Итоговый размер словаря: {tokenizer.get_vocab_size()}")

# if __name__ == "__main__":
#     train_custom_bpe(
#         input_files=["./model/russian_corpus.txt"], 
#         vocab_size=32000, # Оптимально для модели ~100M параметров
#         output_path="./model/tokenizer.json"
#     )