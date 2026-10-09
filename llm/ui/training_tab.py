import flet as ft
import torch
import asyncio
import threading
import os
from queue import Queue

from config import GPT_CONFIG_124M
from traning import GPTTrainer
from dataset import create_dataset


class TrainingTab:
    def __init__(self, app):
        self.app = app
        self.training_thread = None
        self.is_training = False
    
    def build(self):
        # Настройки модели
        self.vocab_size_field = ft.TextField(label="Размер словаря", value=str(GPT_CONFIG_124M["vocab_size"]), width=150, keyboard_type=ft.KeyboardType.NUMBER)
        self.context_length_field = ft.TextField(label="Длина контекста", value=str(GPT_CONFIG_124M["context_length"]), width=150, keyboard_type=ft.KeyboardType.NUMBER)
        self.emb_dim_field = ft.TextField(label="Размер эмбеддинга", value=str(GPT_CONFIG_124M["emb_dim"]), width=150, keyboard_type=ft.KeyboardType.NUMBER)
        
        # Параметры обучения
        self.epochs_field = ft.TextField(label="Эпохи", value="10", width=100, keyboard_type=ft.KeyboardType.NUMBER)
        self.lr_field = ft.TextField(label="Learning Rate", value="0.0005", width=120)
        self.batch_size_field = ft.TextField(label="Batch Size", value="4", width=100, keyboard_type=ft.KeyboardType.NUMBER)
        self.start_context_field = ft.TextField(label="Начальный контекст", value="В начале было", expand=True)
        
        # Пути
        self.corpus_path_field = ft.TextField(label="Путь к корпусу", value="merged_russian_corpus.txt", expand=True)
        self.model_save_path_field = ft.TextField(label="Путь сохранения модели", value="./model/model.pth", expand=True)
        
        # Кнопки
        self.load_data_btn = ft.Button(content="Загрузить данные", icon=ft.Icons.UPLOAD_FILE, on_click=self.on_load_data, bgcolor="#1c3a5a", color="#d2e3fc")
        self.start_btn = ft.Button(content="Начать обучение", icon=ft.Icons.PLAY_ARROW, on_click=self.on_start_training, bgcolor="#1a3a2e", color="#ceead6", disabled=True)
        self.stop_btn = ft.Button(content="Остановить", icon=ft.Icons.STOP, on_click=self.on_stop_training, bgcolor="#5c2b2b", color="#f8d7d7", disabled=True)
        
        # Прогресс и статус
        self.progress = ft.ProgressBar(width=600, visible=False, color="#8ab4f8")
        self.epoch_text = ft.Text("Эпоха: -", size=14, weight=ft.FontWeight.BOLD)
        self.loss_text = ft.Text("Loss: -", size=14, weight=ft.FontWeight.BOLD)
        self.status_text = ft.Text("Готов к обучению", size=14, italic=True, color="#9aa3b2")
        
        # Лог обучения
        self.log_view = ft.ListView(height=220, spacing=5, auto_scroll=True)
        
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Обучение модели", size=24, weight=ft.FontWeight.BOLD),
                    ft.Divider(color="#2a2e36"),
                    
                    ft.Text("Конфигурация модели", size=16, weight=ft.FontWeight.W_500),
                    ft.Row([self.vocab_size_field, self.context_length_field, self.emb_dim_field]),
                    
                    ft.Text("Параметры обучения", size=16, weight=ft.FontWeight.W_500),
                    ft.Row([self.epochs_field, self.lr_field, self.batch_size_field]),
                    self.start_context_field,
                    
                    ft.Text("Пути", size=16, weight=ft.FontWeight.W_500),
                    ft.Row([ft.Text("Корпус:", width=80), self.corpus_path_field]),
                    ft.Row([ft.Text("Модель:", width=80), self.model_save_path_field]),
                    
                    ft.Divider(color="#2a2e36"),
                    ft.Row([self.load_data_btn, self.start_btn, self.stop_btn]),
                    
                    self.progress,
                    ft.Row([self.epoch_text, self.loss_text]),
                    self.status_text,
                    
                    ft.Divider(color="#2a2e36"),
                    ft.Text("Лог обучения", size=16, weight=ft.FontWeight.W_500),
                    self.log_view,
                ],
                scroll=ft.ScrollMode.AUTO,
                spacing=10,
            ),
            padding=20,
            expand=True,
        )
    
    def update_log(self, message: str):
        self.log_view.controls.append(ft.Text(message, size=12, color="#c5c8ce"))
        if len(self.log_view.controls) > 200:
            self.log_view.controls = self.log_view.controls[-100:]
        self.app.page.update()
    
    async def on_load_data(self, e):
        try:
            self.status_text.value = "Загрузка данных"
            self.app.page.update()
            
            cfg = GPT_CONFIG_124M.copy()
            cfg["vocab_size"] = int(self.vocab_size_field.value)
            cfg["context_length"] = int(self.context_length_field.value)
            cfg["emb_dim"] = int(self.emb_dim_field.value)
            
            corpus_path = self.corpus_path_field.value
            self.update_log(f"Загрузка данных из {corpus_path}")
            
            train_loader, valid_loader = await asyncio.to_thread(create_dataset, corpus_path, corpus_path, cfg)
            
            self.app.train_loader = train_loader
            self.app.valid_loader = valid_loader
            self.app.gpt_cfg = cfg
            
            self.update_log(f"Данные загружены. Train: {len(train_loader)} батчей, Val: {len(valid_loader)} батчей")
            self.status_text.value = "Данные загружены. Можно начинать обучение."
            self.start_btn.disabled = False
            self.app.page.update()
            
        except Exception as ex:
            self.update_log(f"Ошибка загрузки данных: {ex}")
            self.status_text.value = f"Ошибка: {ex}"
            self.app.page.update()
    
    async def on_start_training(self, e):
        if self.is_training:
            return
        
        self.is_training = True
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.load_data_btn.disabled = True
        self.progress.visible = True
        self.app.page.update()
        
        try:
            cfg = self.app.gpt_cfg
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.update_log(f"Устройство: {device}")
            
            self.app.trainer = GPTTrainer(cfg, device)
            
            self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
            self.training_thread.start()
            
            await self._monitor_training()
            
        except Exception as ex:
            self.update_log(f"Ошибка запуска обучения: {ex}")
            self.is_training = False
            self._reset_buttons()
            self.app.page.update()
    
    def _training_worker(self):
        try:
            num_epochs = int(self.epochs_field.value)
            lr = float(self.lr_field.value)
            
            self.app.optimizer = self.app.trainer.train(
                train_loader=self.app.train_loader,
                val_loader=self.app.valid_loader,
                num_epochs=num_epochs,
                eval_freq=10,
                eval_iter=5,
                start_context=self.start_context_field.value,
                lr=lr,
                weight_decay=0.1
            )
            
            model_path = self.model_save_path_field.value
            os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
            self.app.trainer.save_checkpoint(model_path, self.app.optimizer)
            self.app.log_queue.put(f"Модель сохранена в {model_path}")
            
        except Exception as ex:
            self.app.log_queue.put(f"Ошибка обучения: {ex}")
        finally:
            self.app.log_queue.put("__TRAINING_DONE__")
    
    async def _monitor_training(self):
        while self.is_training:
            while not self.app.log_queue.empty():
                msg = self.app.log_queue.get()
                if msg == "__TRAINING_DONE__":
                    self.is_training = False
                    self.status_text.value = "Обучение завершено!"
                    self._reset_buttons()
                    self.app.update_plots()
                    self.app.page.update()
                    return
                self.update_log(msg)
            
            if self.app.trainer and self.app.trainer.train_losses:
                last_train = self.app.trainer.train_losses[-1]
                last_val = self.app.trainer.val_losses[-1]
                self.epoch_text.value = f"Шаг: {self.app.trainer.global_step + 1}"
                self.loss_text.value = f"Train: {last_train:.4f} | Val: {last_val:.4f}"
                self.app.update_plots()
            
            self.app.page.update()
            await asyncio.sleep(0.5)
    
    def on_stop_training(self, e):
        self.is_training = False
        self.status_text.value = "Остановка"
        self._reset_buttons()
        self.app.page.update()
    
    def _reset_buttons(self):
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        self.load_data_btn.disabled = False
        self.progress.visible = False