import flet as ft
from queue import Queue


class GPTApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "GPT Trainer - Обучение языковой модели"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.bgcolor = "#101216"
        self.page.dark_theme = ft.Theme(
            scaffold_bgcolor="#101216",
            color_scheme=ft.ColorScheme(
                primary="#8ab4f8",
                on_primary="#062033",
                primary_container="#1c3a5a",
                on_primary_container="#d2e3fc",
                secondary="#81c995",
                on_secondary="#06281a",
                secondary_container="#1a3a2e",
                on_secondary_container="#ceead6",
                surface="#181b22",
                on_surface="#e6e8ec",
                on_surface_variant="#9aa3b2",
                outline="#3a3f4b",
                outline_variant="#2a2e36",
                error="#f2b8b5",
                on_error="#2d0a0a",
                error_container="#5c2b2b",
                surface_container_lowest="#101216",
                surface_container_low="#14171d",
                surface_container="#181b22",
                surface_container_high="#22262f",
                surface_container_highest="#2a2f3a",
            ),
        )
        self.page.theme = self.page.dark_theme
        self.page.window.width = 1200
        self.page.window.height = 800
        
        # Общие данные между вкладками
        self.corpus_path = ft.Ref[ft.TextField]()
        self.tokenizer_path = ft.Ref[ft.TextField]()
        self.model_path = ft.Ref[ft.TextField]()
        
        # Состояние
        self.trainer = None
        self.optimizer = None
        self.train_loader = None
        self.valid_loader = None
        self.is_training = False
        self.log_queue = Queue()
        
        # Построение UI
        self._build_ui()
    
    def _build_ui(self):
        """Построение основного интерфейса"""
        # Импортируем вкладки
        from gui.data_tab import DataTab
        from gui.training_tab import TrainingTab
        from gui.plots_tab import PlotsTab
        from gui.chat_tab import ChatTab
        
        # Создаем вкладки
        self.data_tab = DataTab(self)
        self.training_tab = TrainingTab(self)
        self.plots_tab = PlotsTab(self)
        self.chat_tab = ChatTab(self)
        
        # Основной контейнер с вкладками
        self.tabs = ft.Tabs(
            length=4,
            selected_index=0,
            animation_duration=300,
            expand=True,
            content=ft.Column(
                expand=True,
                spacing=0,
                controls=[
                    ft.TabBar(
                        scrollable=False,
                        tab_alignment=ft.TabAlignment.FILL,
                        indicator_color="#8ab4f8",
                        label_color="#e6e8ec",
                        unselected_label_color="#9aa3b2",
                        divider_color="#2a2e36",
                        tabs=[
                            ft.Tab(label="Данные", icon=ft.Icons.FOLDER),
                            ft.Tab(label="Обучение", icon=ft.Icons.SCHOOL),
                            ft.Tab(label="Графики", icon=ft.Icons.SHOW_CHART),
                            ft.Tab(label="Чат", icon=ft.Icons.CHAT),
                        ],
                    ),
                    ft.TabBarView(
                        expand=True,
                        controls=[
                            self.data_tab.build(),
                            self.training_tab.build(),
                            self.plots_tab.build(),
                            self.chat_tab.build(),
                        ],
                    ),
                ],
            ),
        )
        
        self.page.add(self.tabs)
    
    def log(self, message: str):
        """Добавить сообщение в лог"""
        self.log_queue.put(message)
        # Обновляем лог в training_tab
        self.training_tab.update_log(message)
    
    def update_plots(self):
        """Обновить графики"""
        self.plots_tab.update_plots()


def main(page: ft.Page):
    app = GPTApp(page)
    page.app = app


if __name__ == "__main__":
    ft.run(main)