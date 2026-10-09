import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .acrchecture import GPTModel
from .tokenizator import tokenizer_gpt2

class GPTTrainer:
    def __init__(self, model_config, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = model_config
        
        # Инициализация модели

        self.model = GPTModel(model_config).to(self.device)
        self.tokenizer = tokenizer_gpt2
        
        # Метрики
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
        
        with torch.amp.autocast(dtype=torch.float16):
            logits = self.model(input_batch)
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1), target_batch.flatten()
            )
        return loss

    def train(self, train_loader, val_loader, num_epochs=3, eval_freq=100, eval_iter=10,
              lr=3e-4, weight_decay=0.1, warmup_steps=500, max_grad_norm=1.0,
              gradient_accumulation_steps=16, start_context="### Запрос:\n"):
        """
        Основной цикл обучения с gradient accumulation и mixed precision.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler()
        
        # Learning Rate Scheduler (cosine decay)
        total_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-5
        )
        
        # Warmup scheduler
        def warmup_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        
        self.train_losses, self.val_losses, self.tokens_seen_list = [], [], []
        self.tokens_seen, self.global_step = 0, -1
        best_val_loss = float('inf')
        
        print(f"\nНачало обучения на {num_epochs} эпох.")
        print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
        print(f"Глобальный batch size: {train_loader.batch_size * gradient_accumulation_steps}")
        
        optimizer.zero_grad()
        
        for epoch in range(num_epochs):
            self.model.train()
            accumulated_loss = 0.0
            
            for step, (input_batch, target_batch) in enumerate(train_loader):
                # Forward pass с mixed precision
                loss = self.calc_loss_batch(input_batch, target_batch)
                loss = loss / gradient_accumulation_steps  # Нормализация loss
                
                # Backward pass
                scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                
                # Gradient Accumulation
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Unscale gradients перед clipping
                    scaler.unscale_(optimizer)
                    
                    # Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Learning rate scheduling
                    if self.global_step < warmup_steps:
                        warmup_scheduler.step()
                    else:
                        scheduler.step()
                    
                    self.tokens_seen += input_batch.numel() * gradient_accumulation_steps
                    self.global_step += 1
                    
                    # Периодическая оценка
                    if self.global_step % eval_freq == 0:
                        train_loss = accumulated_loss / gradient_accumulation_steps
                        val_loss = self.evaluate_model(val_loader, eval_iter)
                        
                        self.train_losses.append(train_loss)
                        self.val_losses.append(val_loss)
                        self.tokens_seen_list.append(self.tokens_seen)
                        
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"Ep {epoch+1:02d} (Step {self.global_step:06d}): "
                              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                              f"LR: {current_lr:.2e}")
                        
                        # Сохранение лучшего чекпоинта
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_checkpoint(f"best_model_epoch{epoch+1}.pth", optimizer)
                            print(f"Сохранен лучший чекпоинт (val_loss: {val_loss:.4f})")
                        
                        accumulated_loss = 0.0
            
            # Генерация примера в конце эпохи
            print(f"\nПример генерации после эпохи {epoch+1}:")
            self.generate_text(start_context, max_new_tokens=100, temperature=0.7, top_k=50)
        
        print("\nОбучение завершено!")
        return optimizer

    def evaluate_model(self, val_loader, eval_iter=10):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for i, (input_batch, target_batch) in enumerate(val_loader):
                if i >= eval_iter:
                    break
                loss = self.calc_loss_batch(input_batch, target_batch)
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / min(eval_iter, len(val_loader))

    def generate_text(self, prompt, max_new_tokens=100, temperature=0.7, top_k=50):
        """Генерация текста с поддержкой temperature и top_k."""
        self.model.eval()
        context_size = self.model_config["context_length"]
        idx = self.text_to_token_ids(prompt)
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                with autocast(dtype=torch.float16):
                    logits = self.model(idx_cond)
            logits = logits[:, -1, :]
            
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, 
                                   torch.tensor(float('-inf')).to(self.device), 
                                   logits)
            
            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                
            idx = torch.cat((idx, idx_next), dim=1)
            
        output = self.token_ids_to_text(idx)
        print(f"Prompt: '{prompt}'\nOutput:\n{output}\n{'='*70}")
        self.model.train()
        return output

    def save_checkpoint(self, filepath, optimizer):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "tokens_seen": self.tokens_seen,
            "model_config": self.model_config
        }, filepath)
        print(f"Чекпоинт сохранен: {filepath}")

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.tokens_seen = checkpoint.get("tokens_seen", 0)
        self.model.eval()
        print(f"Чекпоинт загружен: {filepath}")