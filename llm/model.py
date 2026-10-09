import torch
from traning import GPTTrainer
from dataset import create_dataset
from config import GPT_CONFIG_124M

if __name__ == "__main__":
    # Конфигурация модели
    gpt_cfg = GPT_CONFIG_124M
    
    # ВАЖНО: Убедитесь, что в config.py обновлены параметры:
    # "vocab_size": 32000  (размер вашего BPE словаря)
    # "context_length": 512 (благодаря RoPE)
    
    print("=" * 70)
    print("ЗАПУСК ОБУЧЕНИЯ РУССКОЯЗЫЧНОЙ МОДЕЛИ")
    print("=" * 70)
    print(f"Конфигурация модели:")
    print(f"  - Размер словаря: {gpt_cfg['vocab_size']}")
    print(f"  - Длина контекста: {gpt_cfg['context_length']}")
    print(f"  - Размерность эмбеддинга: {gpt_cfg['emb_dim']}")
    print(f"  - Количество слоев: {gpt_cfg['n_layers']}")
    print(f"  - Количество голов внимания: {gpt_cfg['n_heads']}")
    print("=" * 70)
    
    # Загрузка данных (используем объединенный русский корпус)
    print("\nЗагрузка данных")
    corpus_file = "./model/russian_corpus.txt"  # Файл из prepare_dataset.py
    
    try:
        train_loader, valid_loader = create_dataset(corpus_file, corpus_file, gpt_cfg)
        print(f"Данные успешно загружены из {corpus_file}")
    except FileNotFoundError:
        print(f"Файл {corpus_file} не найден!")
        print("   Запустите сначала prepare_dataset.py для создания корпуса.")
        exit(1)
    
    # Выбор устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nУстройство для обучения: {device}")
    
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Доступно памяти: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Инициализация тренера
    print("\nИнициализация модели")
    model_trainer = GPTTrainer(gpt_cfg, device)
    
    # Запуск обучения
    print("\nНачало обучения")
    optimizer = model_trainer.train(
        train_loader=train_loader,
        val_loader=valid_loader,
        num_epochs=5,           # Количество эпох
        eval_freq=50,            # Оценка каждые 50 шагов
        eval_iter=5,             # Количество батчей для оценки
        start_context="В начале было",  # Русский промпт для генерации
        lr=5e-4,                 # Скорость обучения
        weight_decay=0.1         # Затухание весов
    )
    
    # Тестирование генерации на русском языке
    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ")
    print("=" * 70)
    
    test_prompts = [
        "В начале было",
        "Искусственный интеллект это",
        "Сегодня я решил",
        "Наука и технологии"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Промпт: '{prompt}'")
        model_trainer.generate_text(
            prompt, 
            max_new_tokens=50, 
            temperature=0.8, 
            top_k=40
        )
    
    # Финальная оценка модели
    print("\n" + "=" * 70)
    print("ФИНАЛЬНАЯ ОЦЕНКА")
    print("=" * 70)
    
    train_loss, val_loss = model_trainer.evaluate_model(train_loader, valid_loader, eval_iter=10)
    print(f"Финальный Train Loss: {train_loss:.4f}")
    print(f"Финальный Val Loss: {val_loss:.4f}")
    
    # Вывод детальной статистики обучения
    model_trainer.print_training_summary()
    
    # Визуализация потерь
    print("\nПостроение графиков")
    model_trainer.plot_losses()
    
    # Сохранение модели
    print("\nСохранение модели")
    import os
    os.makedirs("./model", exist_ok=True)
    model_trainer.save_checkpoint("./model/model.pth", optimizer)
    
    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("=" * 70)
    print(f"Модель сохранена в: ./model/model.pth")
    print(f"Токенизатор: russian_bpe_tokenizer.json")
    print(f"Корпус данных: {corpus_file}")
    print("=" * 70)