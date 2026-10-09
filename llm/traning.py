import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from config import GPT_CONFIG_124M
from acrchecture import GPTModel
from tokenizator import tokenizer_gpt2

class GPTTrainer:
    def __init__(self, model_config=GPT_CONFIG_124M, device=None):
        """
        Инициализация тренера для GPT модели.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = model_config
        self.tokenizer = tokenizer_gpt2
        
        # Инициализация модели
        self.model = GPTModel(self.model_config)
        self.model.to(self.device)
        
        # Метрики для отслеживания обучения
        self.train_losses = []
        self.val_losses = []
        self.tokens_seen_list = []
        self.tokens_seen = 0
        self.global_step = -1
        
        print(f"Модель инициализирована. Устройство: {self.device}")
        print(f"Параметры модели: {sum(p.numel() for p in self.model.parameters()):,}")

    def text_to_token_ids(self, text):
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        return torch.tensor(encoded).unsqueeze(0).to(self.device)

    def token_ids_to_text(self, token_ids):
        flat = token_ids.squeeze(0)
        return self.tokenizer.decode(flat.tolist())

    def calc_loss_batch(self, input_batch, target_batch):
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
        )
        return loss

    def calc_loss_loader(self, data_loader, num_batches=None):
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        
        num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
        
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(input_batch, target_batch)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

    def evaluate_model(self, train_loader, val_loader, eval_iter: int = 5):
        self.model.eval()
        with torch.no_grad():
            train_loss = self.calc_loss_loader(train_loader, num_batches=eval_iter)
            val_loss = self.calc_loss_loader(val_loader, num_batches=eval_iter)
        self.model.train()
        return train_loss, val_loss

    def train(self, train_loader, val_loader, num_epochs=10, eval_freq=5, eval_iter=5, 
              start_context="Every effort moves you", lr=5e-4, weight_decay=0.1):
        """
        Основной цикл обучения модели.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.train_losses, self.val_losses, self.tokens_seen_list = [], [], []
        self.tokens_seen, self.global_step = 0, -1
        
        print(f"\nНачало обучения на {num_epochs} эпох.")
        
        for epoch in range(num_epochs):
            self.model.train()
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()
                loss = self.calc_loss_batch(input_batch, target_batch)
                loss.backward()
                optimizer.step()
                
                self.tokens_seen += input_batch.numel()
                self.global_step += 1
                
                # Периодическая оценка
                if self.global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model(train_loader, val_loader, eval_iter)
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)
                    self.tokens_seen_list.append(self.tokens_seen)
                    print(f"Ep {epoch+1:02d} (Step {self.global_step:06d}): "
                          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Генерация примера в конце эпохи
            print(f"\nПример генерации после эпохи {epoch+1}:")
            self.generate_text(start_context, max_new_tokens=40, temperature=0.8, top_k=30)
            
        print("Обучение завершено!")
        return optimizer

    def print_training_summary(self):
        """Вывод детальной оценки пройденного обучения."""
        print("ИТОГОВАЯ ОЦЕНКА ОБУЧЕНИЯ")
        
        if not self.train_losses:
            print("Обучение не было проведено, статистика отсутствует.")
            return
            
        initial_train, final_train = self.train_losses[0], self.train_losses[-1]
        initial_val, final_val = self.val_losses[0], self.val_losses[-1]
        
        train_improvement = ((initial_train - final_train) / initial_train) * 100
        val_improvement = ((initial_val - final_val) / initial_val) * 100
        
        print(f"Начальный Train Loss:  {initial_train:.4f}")
        print(f"Конечный Train Loss:    {final_train:.4f} (Улучшение: {train_improvement:.2f}%)")
        print(f"Начальный Val Loss:     {initial_val:.4f}")
        print(f"Конечный Val Loss:      {final_val:.4f} (Улучшение: {val_improvement:.2f}%)")
        print(f"Всего обработано токенов: {self.tokens_seen:,}")
        print(f"Всего шагов оптимизации:  {self.global_step + 1}")
        
        # Проверка на переобучение
        if final_val > initial_val:
            print("\nОбнаружены признаки переобучения!")
            print("   (Конечный Val Loss выше начального. Попробуйте увеличить dropout или уменьшить lr)")
        elif (final_val - final_train) > 1.0:
            print("\nБольшой разрыв между Train и Val loss.")
            print("   (Возможно переобучение на тренировочных данных)")
        else:
            print("\nМодель показывает здоровую динамику обучения.")
            
        print("=" * 70)
        return {"train": {"value": final_train, "percent": train_improvement}, "test": {"value": final_val, "percent": val_improvement}}

    def generate_text(self, prompt, max_new_tokens=25, temperature=0.0, top_k=None):
        """Генерация текста с поддержкой temperature и top_k."""
        self.model.eval()
        context_size = self.model_config["context_length"]
        idx = self.text_to_token_ids(prompt)
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self.model(idx_cond)
            logits = logits[:, -1, :]
            
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(self.device), logits)
            
            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                
            idx = torch.cat((idx, idx_next), dim=1)
            
        print(f"Prompt: '{prompt}'\nOutput: {self.token_ids_to_text(idx)}")
        self.model.train()
        return self.token_ids_to_text(idx)

    def plot_losses(self):
        """Визуализация кривых обучения."""
        if not self.train_losses:
            return
            
        fig, ax1 = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(self.train_losses) + 1)
        
        ax1.plot(epochs, self.train_losses, label="Training loss", marker='o', linewidth=2)
        ax1.plot(epochs, self.val_losses, label="Validation loss", marker='s', linestyle='-.', linewidth=2)
        ax1.set_xlabel("Evaluation Steps")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twiny()
        ax2.plot(self.tokens_seen_list, self.train_losses, alpha=0)
        ax2.set_xlabel("Tokens Seen")
        
        plt.title("Training and Validation Loss Over Time")
        fig.tight_layout()
        plt.show()

    def save_checkpoint(self, filepath, optimizer):
        """Сохранение модели, оптимизатора и истории обучения."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "tokens_seen": self.tokens_seen,
            "model_config": self.model_config
        }, filepath)
        print(f"Чекпоинт успешно сохранен в: {filepath}")

    def load_checkpoint(self, filepath):
        """Загрузка модели и истории обучения."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.tokens_seen = checkpoint.get("tokens_seen", 0)
        self.model.eval()
        print(f"Чекпоинт успешно загружен из: {filepath}")