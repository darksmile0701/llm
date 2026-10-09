import flet as ft
import matplotlib
matplotlib.use('Agg')  # Не-GUI backend для matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import base64


class PlotsTab:
    def __init__(self, app):
        self.app = app
        self.plot_image = ft.Image(
            src="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            width=800,
            height=400,
        )
        self.plot_box = ft.Container(
            content=self.plot_image,
            alignment=ft.Alignment.CENTER,
            bgcolor="#181b22",
            border_radius=8,
            padding=12,
            visible=False,
        )
   
    def build(self):
        """Построение вкладки графиков"""
        
        self.refresh_btn = ft.Button(
            content="Обновить графики",
            icon=ft.Icons.REFRESH,
            bgcolor="#1c3a5a",
            color="#d2e3fc",
            on_click=self.on_refresh,
        )
        
        self.status_text = ft.Text("Нет данных для отображения", size=14, italic=True, color="#9aa3b2")
        
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Графики обучения", size=24, weight=ft.FontWeight.BOLD),
                    ft.Divider(color="#2a2e36"),
                    ft.Row([self.refresh_btn]),
                    self.status_text,
                    self.plot_box,
                ],
                scroll=ft.ScrollMode.AUTO,
                spacing=10,
            ),
            padding=20,
            expand=True,
        )
    
    def on_refresh(self, e):
        """Обновление графиков"""
        self.update_plots()
    
    def update_plots(self):
        """Обновление графиков из данных тренера"""
        if not self.app.trainer or not self.app.trainer.train_losses:
            self.status_text.value = "Нет данных для отображения. Начните обучение."
            self.app.page.update()
            return
        
        try:
            # Создаем фигуру matplotlib
            plt.style.use("dark_background")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor("#181b22")
            
            # График 1: Loss
            epochs = range(1, len(self.app.trainer.train_losses) + 1)
            ax1.plot(epochs, self.app.trainer.train_losses, 
                    label="Training loss", marker='o', linewidth=2, color='#8ab4f8')
            ax1.plot(epochs, self.app.trainer.val_losses, 
                    label="Validation loss", marker='s', linestyle='-.', 
                    linewidth=2, color='#fdd663')
            ax1.set_facecolor("#181b22")
            ax1.set_xlabel("Evaluation Steps")
            ax1.set_ylabel("Loss")
            ax1.set_title("Loss during training")
            ax1.legend()
            ax1.grid(True, alpha=0.25, color="#3a3f4b")
            
            # График 2: Loss vs Tokens seen
            ax2.set_facecolor("#181b22")
            if self.app.trainer.tokens_seen_list:
                ax2.plot(self.app.trainer.tokens_seen_list, 
                        self.app.trainer.train_losses,
                        label="Training loss", marker='o', linewidth=2, color='#81c995')
                ax2.plot(self.app.trainer.tokens_seen_list, 
                        self.app.trainer.val_losses,
                        label="Validation loss", marker='s', linestyle='-.', 
                        linewidth=2, color='#f2b8b5')
                ax2.set_xlabel("Tokens Seen")
                ax2.set_ylabel("Loss")
                ax2.set_title("Loss vs Tokens Seen")
                ax2.legend()
                ax2.grid(True, alpha=0.25, color="#3a3f4b")
            
            plt.tight_layout()
            
            # Конвертируем в PNG
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
            buf.seek(0)
            plt.close(fig)
            
            # Кодируем в base64
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Обновляем изображение в UI
            self.plot_image.src = img_base64
            self.plot_box.visible = True
            self.status_text.value = (
                f"Шагов: {len(self.app.trainer.train_losses)} | "
                f"Последний Train: {self.app.trainer.train_losses[-1]:.4f} | "
                f"Последний Val: {self.app.trainer.val_losses[-1]:.4f}"
            )
            self.app.page.update()
            
        except Exception as ex:
            self.status_text.value = f"Ошибка построения графика: {ex}"
            self.app.page.update()