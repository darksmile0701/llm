import flet as ft
import os
import asyncio

from llm import prepare_russian_corpus, train_custom_bpe


class DataTab:
    def __init__(self, app):
        self.app = app
    
    def build(self):
        # Поля для путей
        self.corpus_field = ft.TextField(
            label="Путь к файлу корпуса",
            value="merged_russian_corpus.txt",
            expand=True,
            read_only=True,
        )
        
        self.tokenizer_field = ft.TextField(
            label="Путь для сохранения токенизатора",
            value="russian_bpe_tokenizer.json",
            expand=True,
            read_only=True,
        )
        
        # Настройки подготовки данных
        self.cb_wikipedia = ft.Checkbox(label="Wikipedia RU", value=True)
        self.cb_gazeta = ft.Checkbox(label="Газета", value=False)
        self.cb_pikabu = ft.Checkbox(label="Pikabu", value=False)
        
        self.max_samples_field = ft.TextField(
            label="Макс. образцов на источник",
            value="50000",
            width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        
        self.vocab_size_field = ft.TextField(
            label="Размер словаря токенизатора",
            value="32000",
            width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        
        # Кнопки действий
        self.download_btn = ft.Button(
            content="Скачать и подготовить датасет",
            icon=ft.Icons.DOWNLOAD,
            on_click=self.on_download_corpus,
            bgcolor="#1c3a5a",
            color="#d2e3fc",
        )
        
        self.train_tokenizer_btn = ft.Button(
            content="Обучить токенизатор",
            icon=ft.Icons.BUILD,
            on_click=self.on_train_tokenizer,
            bgcolor="#1a3a2e",
            color="#ceead6",
        )
        
        # Прогресс и статус
        self.progress = ft.ProgressBar(width=500, visible=False, color="#8ab4f8")
        self.status_text = ft.Text("", size=14, color="#9aa3b2")
        
        # Лог
        self.log_view = ft.ListView(
            height=220,
            spacing=5,
            auto_scroll=True,
        )
        
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Подготовка данных", size=24, weight=ft.FontWeight.BOLD),
                    ft.Divider(color="#2a2e36"),
                    
                    ft.Text("1. Выбор путей", size=16, weight=ft.FontWeight.W_500),
                    ft.Row([
                        ft.IconButton(
                            icon=ft.Icons.FOLDER_OPEN,
                            icon_color="#8ab4f8",
                            tooltip="Выбрать путь к корпусу",
                            on_click=self.pick_corpus,
                        ),
                        self.corpus_field,
                    ]),
                    ft.Row([
                        ft.IconButton(
                            icon=ft.Icons.FOLDER_OPEN,
                            icon_color="#8ab4f8",
                            tooltip="Выбрать путь к токенизатору",
                            on_click=self.pick_tokenizer,
                        ),
                        self.tokenizer_field,
                    ]),
                    
                    ft.Divider(color="#2a2e36"),
                    ft.Text("2. Настройки", size=16, weight=ft.FontWeight.W_500),
                    ft.Row([self.cb_wikipedia, self.cb_gazeta, self.cb_pikabu]),
                    ft.Row([self.max_samples_field, self.vocab_size_field]),
                    
                    ft.Divider(color="#2a2e36"),
                    ft.Text("3. Действия", size=16, weight=ft.FontWeight.W_500),
                    ft.Row([self.download_btn, self.train_tokenizer_btn]),
                    self.progress,
                    self.status_text,
                    
                    ft.Divider(color="#2a2e36"),
                    ft.Text("Лог операций", size=16, weight=ft.FontWeight.W_500),
                    self.log_view,
                ],
                scroll=ft.ScrollMode.AUTO,
                spacing=10,
            ),
            padding=20,
            expand=True,
        )
    
    async def pick_corpus(self, e):
        path = await ft.FilePicker().save_file(
            dialog_title="Сохранить корпус как",
            file_name="merged_russian_corpus.txt",
        )
        if path:
            self.corpus_field.value = path
            self.app.page.update()
    
    async def pick_tokenizer(self, e):
        path = await ft.FilePicker().save_file(
            dialog_title="Сохранить токенизатор как",
            file_name="russian_bpe_tokenizer.json",
        )
        if path:
            self.tokenizer_field.value = path
            self.app.page.update()
    
    def add_log(self, message: str):
        self.log_view.controls.append(ft.Text(message, size=12, color="#c5c8ce"))
        self.app.page.update()
    
    async def on_download_corpus(self, e):
        self.progress.visible = True
        self.download_btn.disabled = True
        self.app.page.update()
        
        try:
            sources = []
            if self.cb_wikipedia.value: sources.append("wikipedia")
            if self.cb_gazeta.value: sources.append("gazeta")
            if self.cb_pikabu.value: sources.append("pikabu")
            
            if not sources:
                self.add_log("❌ Выберите хотя бы один источник данных!")
                return
            
            max_samples = int(self.max_samples_field.value)
            output_file = self.corpus_field.value
            
            self.add_log(f"🚀 Начало подготовки корпуса: {sources}")
            self.status_text.value = "Скачивание и обработка данных..."
            self.app.page.update()
            
            await asyncio.to_thread(
                prepare_russian_corpus,
                output_file=output_file,
                max_samples=max_samples,
                sources=sources
            )
            
            self.add_log(f"✅ Корпус успешно сохранен в: {output_file}")
            self.status_text.value = "Готово!"
            
        except Exception as ex:
            self.add_log(f"❌ Ошибка: {ex}")
            self.status_text.value = f"Ошибка: {ex}"
        finally:
            self.progress.visible = False
            self.download_btn.disabled = False
            self.app.page.update()
    
    async def on_train_tokenizer(self, e):
        self.progress.visible = True
        self.train_tokenizer_btn.disabled = True
        self.app.page.update()
        
        try:
            corpus_file = self.corpus_field.value
            tokenizer_path = self.tokenizer_field.value
            vocab_size = int(self.vocab_size_field.value)
            
            if not os.path.exists(corpus_file):
                raise FileNotFoundError(f"Файл корпуса не найден: {corpus_file}")
            
            self.add_log(f"🔧 Обучение токенизатора на {corpus_file}")
            self.status_text.value = "Обучение токенизатора..."
            self.app.page.update()
            
            await asyncio.to_thread(
                train_custom_bpe,
                output_tokenizer_path=tokenizer_path,
                corpus_file=corpus_file,
                vocab_size=vocab_size
            )
            
            self.add_log(f"✅ Токенизатор сохранен в: {tokenizer_path}")
            self.status_text.value = "Готово!"
            
        except Exception as ex:
            self.add_log(f"❌ Ошибка: {ex}")
            self.status_text.value = f"Ошибка: {ex}"
        finally:
            self.progress.visible = False
            self.train_tokenizer_btn.disabled = False
            self.app.page.update()