# import tiktoken

# tokenizer_gpt2 = tiktoken.get_encoding("gpt2")

from tokenizers import Tokenizer

# Загружаем обученный токенизатор
tokenizer = Tokenizer.from_file("./model/tokenizer.json")

# Обертки для совместимости с вашим кодом
class CustomTokenizerWrapper:
    def __init__(self, hf_tokenizer):
        self.hf_tokenizer = hf_tokenizer

    def encode(self, text, allowed_special=None):
        # Возвращает список ID
        return self.hf_tokenizer.encode(text).ids

    def decode(self, ids):
        # Убираем специальные токены при декодировании, если нужно
        return self.hf_tokenizer.decode(ids, skip_special_tokens=True)

tokenizer_gpt2 = CustomTokenizerWrapper(tokenizer)