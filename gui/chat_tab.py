import flet as ft
import torch
import asyncio

from llm import GPT_CONFIG_124M, GPTModel, tokenizer_gpt2


class ChatTab:
    def __init__(self, app):
        self.app = app
        self.model = None
        self.device = None
    
    def build(self):
        """Построение вкладки чата"""
        
        # Настройки
        self.model_path_field = ft.TextField(

            label="Путь к модели",
            value="./model/model.pth",
            expand=True,
        )
        
        self.load_model_btn = ft.Button(
            content="Загрузить модель",
            icon=ft.Icons.DOWNLOAD,
            bgcolor="#1c3a5a",
            color="#d2e3fc",
            on_click=self.on_load_model,
        )
        
        # Параметры генерации
        self.max_tokens_field = ft.TextField(
            label="Макс. токенов",
            value="100",
            width=120,
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        
        self.temperature_field = ft.TextField(
            label="Temperature",
            value="0.8",
            width=120,
        )
        
        self.top_k_field = ft.TextField(
            label="Top-K",
            value="40",
            width=120,
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        
        # Чат
        self.chat_view = ft.ListView(
            expand=True,
            spacing=10,
            auto_scroll=True,
        )
        
        self.prompt_field = ft.TextField(
            label="Введите сообщение",
            hint_text="Начните диалог...",
            expand=True,
            on_submit=self.on_send,
        )
        
        self.send_btn = ft.IconButton(
            icon=ft.Icons.SEND,
            icon_color="#8ab4f8",
            on_click=self.on_send,
            icon_size=30,
        )
        
        self.clear_btn = ft.IconButton(
            icon=ft.Icons.DELETE,
            icon_color="#f2b8b5",
            on_click=self.on_clear,
            tooltip="Очистить чат",
        )
        
        self.status_text = ft.Text("Модель не загружена", size=14, italic=True, color="#9aa3b2")
        
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Чат с моделью", size=24, weight=ft.FontWeight.BOLD),
                    ft.Divider(color="#2a2e36"),
                    
                    ft.Row([
                        self.model_path_field,
                        self.load_model_btn,
                    ]),
                    
                    ft.Row([
                        self.max_tokens_field,
                        self.temperature_field,
                        self.top_k_field,
                    ]),
                    
                    self.status_text,
                    
                    ft.Container(
                        content=self.chat_view,
                        expand=True,
                        bgcolor="#181b22",
                        border=ft.Border.all(1, "#3a3f4b"),
                        border_radius=10,
                        padding=10,
                    ),
                    
                    ft.Row([
                        self.prompt_field,
                        self.send_btn,
                        self.clear_btn,
                    ]),
                ],
                spacing=10,
                expand=True,
            ),
            padding=20,
            expand=True,
        )
    
    async def on_load_model(self, e):
        """Загрузка модели"""
        try:
            self.status_text.value = "Загрузка модели..."
            self.app.page.update()
            
            model_path = self.model_path_field.value
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Загружаем чекпоинт
            checkpoint = await asyncio.to_thread(
                torch.load, model_path, map_location=self.device
            )
            
            # Получаем конфиг
            if isinstance(checkpoint, dict) and "model_config" in checkpoint:
                cfg = checkpoint["model_config"]
            else:
                cfg = GPT_CONFIG_124M
            
            # Создаем модель
            self.model = GPTModel(cfg)
            
            # Загружаем веса
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.status_text.value = f"✅ Модель загружена на {self.device}"
            self.app.page.update()
            
        except Exception as ex:
            self.status_text.value = f"❌ Ошибка загрузки: {ex}"
            self.app.page.update()
    
    async def on_send(self, e):
        """Отправка сообщения"""
        prompt = self.prompt_field.value.strip()
        if not prompt:
            return
        
        if self.model is None:
            self._add_message("Система", "⚠️ Сначала загрузите модель!")
            return
        
        # Добавляем сообщение пользователя
        self._add_message("Вы", prompt)
        self.prompt_field.value = ""
        self.app.page.update()
        
        # Генерируем ответ
        try:
            self.status_text.value = "🤔 Генерация ответа..."
            self.app.page.update()
            
            max_tokens = int(self.max_tokens_field.value)
            temperature = float(self.temperature_field.value)
            top_k = int(self.top_k_field.value)
            
            # Используем метод generate_text из trainer, если доступен
            if self.app.trainer:
                response = await asyncio.to_thread(
                    self.app.trainer.generate_text,
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
            else:
                # Генерируем напрямую
                response = await asyncio.to_thread(
                    self._generate,
                    prompt, max_tokens, temperature, top_k
                )
            
            self._add_message("Модель", response)
            self.status_text.value = "✅ Готово"
            
        except Exception as ex:
            self._add_message("Ошибка", str(ex))
            self.status_text.value = f"❌ Ошибка: {ex}"
        
        self.app.page.update()
    
    def _generate(self, prompt, max_tokens, temperature, top_k):
        """Прямая генерация текста"""
        context_size = self.model.pos_emb.weight.shape[0] if hasattr(self.model, 'pos_emb') else 512
        
        # Токенизация
        encoded = tokenizer_gpt2.encode(prompt, allowed_special={'<|endoftext|>'})
        idx = torch.tensor(encoded).unsqueeze(0).to(self.device)
        
        # Генерация
        for _ in range(max_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self.model(idx_cond)
            logits = logits[:, -1, :]
            
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float('-inf')).to(self.device),
                    logits
                )
            
            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        # Декодирование
        flat = idx.squeeze(0)
        return tokenizer_gpt2.decode(flat.tolist())
    
    def _add_message(self, sender: str, text: str):
        """Добавить сообщение в чат"""
        is_user = sender == "Вы"
        name_color = "#8ab4f8" if is_user else "#81c995"
        bubble = ft.Container(
            content=ft.Column([
                ft.Text(sender, weight=ft.FontWeight.BOLD, color=name_color, size=14),
                ft.Text(text, size=14, color="#e6e8ec"),
            ]),
            bgcolor="#1c3a5a" if is_user else "#1a3a2e",
            padding=10,
            border_radius=10,
            width=520,
        )
        self.chat_view.controls.append(
            ft.Row(
                [bubble],
                alignment=ft.MainAxisAlignment.START if is_user else ft.MainAxisAlignment.END,
            )
        )
    
    def on_clear(self, e):
        """Очистить чат"""
        self.chat_view.controls.clear()
        self.app.page.update()