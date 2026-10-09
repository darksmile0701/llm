import re
from datasets import load_dataset, concatenate_datasets

def clean_text(text: str) -> str:
    """Базовая очистка текста от мусора"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text.strip()

def load_wikipedia_ru(max_samples: int = 100000):
    """Загрузка русской Википедии (используем более надежный источник)"""
    print(f"Загрузка Wikipedia RU (лимит: {max_samples} статей)...")
    try:
        # Используем актуальную версию русской Википедии
        dataset = load_dataset(
            "wikimedia/wikipedia", 
            "20231101.ru",  # Более свежая и стабильная версия
            split=f"train[:{max_samples}]",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Не удалось загрузить wikipedia: {e}")
        print("Пробуем альтернативный источник...")
        # Альтернатива: готовый датасет русской Википедии
        dataset = load_dataset(
            "IlyaGusev/wikitext_100m",
            split=f"train[:{max_samples}]"
        )
    
    def format_example(example):
        title = clean_text(example.get("title", ""))
        text = clean_text(example.get("text", ""))
        if len(text) > 100:
            return {"text": f"Заголовок: {title}\n{text}"}
        return {"text": ""}
    
    dataset = dataset.map(format_example).filter(lambda x: len(x["text"]) > 100)
    print(f"Wikipedia RU: загружено {len(dataset)} статей")
    return dataset

def load_gazeta(max_samples: int = 100000):
    """Загрузка новостного корпуса Газета"""
    print(f"Загрузка Gazeta (лимит: {max_samples} статей)...")
    try:
        dataset = load_dataset(
            "IlyaGusev/gazeta", 
            split=f"train[:{max_samples}]",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Не удалось загрузить gazeta: {e}")
        return None
    
    def format_example(example):
        title = clean_text(example.get("title", ""))
        text = clean_text(example.get("text", ""))
        if len(text) > 100:
            return {"text": f"Новость: {title}\n{text}"}
        return {"text": ""}
    
    dataset = dataset.map(format_example).filter(lambda x: len(x["text"]) > 100)
    print(f"Gazeta: загружено {len(dataset)} статей")
    return dataset

# def load_books(max_samples: int = 100000):
#     print(f"Загрузка книг (лимит: {max_samples})...")
#     dataset = load_dataset("your_dataset_name", split=f"train[:{max_samples}]")
    
#     def format_example(example):
#         text = clean_text(example.get("text", ""))
#         if len(text) > 100:
#             return {"text": f"Книга:\n{text}"}
#         return {"text": ""}
    
#     dataset = dataset.map(format_example).filter(lambda x: len(x["text"]) > 100)
#     print(f"Книги: загружено {len(dataset)} документов")
#     return dataset

def load_pikabu(max_samples: int = 100000):
    """Загрузка корпуса Пикабу (разговорный стиль)"""
    print(f"Загрузка Pikabu (лимит: {max_samples} постов)...")
    try:
        dataset = load_dataset(
            "IlyaGusev/pikabu", 
            split=f"train[:{max_samples}]",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Не удалось загрузить pikabu: {e}")
        return None
    
    def format_example(example):
        title = clean_text(example.get("title", ""))
        text = clean_text(example.get("text", ""))
        if len(text) > 100:
            return {"text": f"Пост: {title}\n{text}"}
        return {"text": ""}
    
    dataset = dataset.map(format_example).filter(lambda x: len(x["text"]) > 100)
    print(f"Pikabu: загружено {len(dataset)} постов")
    return dataset

def merge_datasets(datasets_list: list, shuffle: bool = True, seed: int = 42):
    """Объединение нескольких датасетов"""
    # Убираем None значения (если какой-то датасет не загрузился)
    datasets_list = [ds for ds in datasets_list if ds is not None]
    
    if not datasets_list:
        print("Нет доступных датасетов для объединения!")
        return None
    
    print(f"Объединение {len(datasets_list)} датасетов...")
    merged = concatenate_datasets(datasets_list)
    
    if shuffle:
        print(f"Перемешивание данных (seed={seed})...")
        merged = merged.shuffle(seed=seed)
    
    print(f"Итого объединено: {len(merged)} документов")
    return merged

def prepare_russian_corpus(output_file: str = "russian_corpus_clean.txt", 
                           max_samples: int = 100000,
                           sources: list = None):
    """
    Подготовка русского корпуса данных
    """
    if sources is None:
        sources = ['wikipedia']
    
    print(f"Начало подготовки корпуса. Источники: {sources}")
    print(f"Лимит на каждый источник: {max_samples} записей")
    print("-" * 60)
    
    datasets_to_merge = []
    
    # Загружаем запрошенные источники
    if 'wikipedia' in sources:
        ds = load_wikipedia_ru(max_samples)
        if ds is not None:
            datasets_to_merge.append(ds)
    
    if 'gazeta' in sources:
        ds = load_gazeta(max_samples)
        if ds is not None:
            datasets_to_merge.append(ds)
    
    if 'pikabu' in sources:
        ds = load_pikabu(max_samples)
        if ds is not None:
            datasets_to_merge.append(ds)

    if 'books' in sources:
        ds = load_books(max_samples)
        if ds is not None:
            datasets_to_merge.append(ds)
    
    if not datasets_to_merge:
        print("Не удалось загрузить ни одного источника!")
        return
    
    # Объединяем датасеты
    if len(datasets_to_merge) > 1:
        final_dataset = merge_datasets(datasets_to_merge, shuffle=True)
    else:
        final_dataset = datasets_to_merge[0]
    
    if final_dataset is None:
        print("Не удалось создать финальный датасет!")
        return
    
    # Сохраняем в файл
    print(f"Сохранение в {output_file}...")
    valid_docs = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in final_dataset:
            f.write(item["text"] + "\n\n")
            valid_docs += 1
    
    print(f"Готово! Сохранено {valid_docs} качественных документов в {output_file}")

if __name__ == "__main__":
    # Для теста берем 50 000 статей
    prepare_russian_corpus(
        output_file="./model/russian_corpus.txt",
        max_samples=10000,
        sources=['wikipedia', 'gazeta', 'pikabu']
    )