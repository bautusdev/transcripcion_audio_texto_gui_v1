import sys
import os

# Silenciar mensajes de terminal y forzar el uso exclusivo del modelo local.
# IMPORTANTE: estas variables deben establecerse ANTES de importar transformers.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import torch
# IMPORTANTE: importar transformers en el hilo principal.
# Al importarlo (y a triton/torch._dynamo) dentro de un QThread se produce
# un segfault al cargar las extensiones C de triton fuera del hilo principal.
from transformers import pipeline
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

import numpy as np
import librosa

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFileDialog, QStatusBar, QSizePolicy,
    QProgressBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor


# El modelo se carga desde la carpeta local (sin descargas de internet).
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_whisper_small")

# Tamaño de cada trozo de audio que se envía a Whisper (en segundos).
# Whisper trabaja internamente con ventanas de 30 s; trocear nos permite
# procesar audios de cualquier duración y reportar progreso real.
SAMPLE_RATE = 16000
CHUNK_SECONDS = 30

# Cache del pipeline: cargar el modelo es costoso (varios segundos),
# así que se hace una sola vez y se reutiliza en cada transcripción.
_pipe = None


def get_pipe():
    """Devuelve el pipeline de ASR, creándolo (y cacheándolo) si es necesario."""
    global _pipe
    if _pipe is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_DIR,
            device=device,
        )
    return _pipe


class TranscriptionWorker(QThread):
    """Hilo de trabajo para no congelar la interfaz durante la transcripción.

    El audio se divide en trozos de CHUNK_SECONDS; por cada trozo se emite
    el texto parcial y el progreso, de modo que la interfaz siempre muestra
    actividad incluso con archivos largos.
    """
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)
    progress = pyqtSignal(int)   # 0-100
    chunk_text = pyqtSignal(str)  # texto de cada trozo, para mostrarlo en vivo

    def __init__(self, audio_path, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path

    def run(self):
        try:
            if not os.path.isdir(MODEL_DIR):
                self.failed.emit(
                    f"No se encontró el modelo local en:\n{MODEL_DIR}\n\n"
                    "Ejecuta primero: python descargar_modelo.py"
                )
                return

            first_load = _pipe is None
            self.status.emit("Cargando modelo local..." if first_load
                             else "Preparando transcripción...")
            asr_pipe = get_pipe()

            # Decodificar TODO el audio a 16 kHz mono (ocupa ~64 KB por
            # segundo de audio, así que incluso horas caben en memoria).
            self.status.emit("Leyendo archivo de audio...")
            audio, _ = librosa.load(self.audio_path, sr=SAMPLE_RATE, mono=True)
            if audio.size == 0:
                self.failed.emit("El archivo de audio está vacío o no se pudo decodificar.")
                return

            duration = len(audio) / SAMPLE_RATE
            chunk_len = CHUNK_SECONDS * SAMPLE_RATE
            total_chunks = max(1, int(np.ceil(len(audio) / chunk_len)))
            self.status.emit(
                f"Transcribiendo {self._fmt_time(duration)} de audio "
                f"({total_chunks} parte{'s' if total_chunks > 1 else ''})..."
            )

            texts = []
            for i in range(total_chunks):
                chunk = audio[i * chunk_len:(i + 1) * chunk_len]
                result = asr_pipe(
                    chunk,
                    generate_kwargs={"language": "spanish", "task": "transcribe"},
                )
                text = result["text"].strip()
                if text:
                    texts.append(text)
                    self.chunk_text.emit(text)

                pct = int((i + 1) / total_chunks * 100)
                self.progress.emit(pct)
                self.status.emit(
                    f"Transcribiendo... {pct}% "
                    f"(parte {i + 1} de {total_chunks})"
                )

            self.finished_ok.emit(" ".join(texts))
        except Exception as e:
            self.failed.emit(str(e))

    @staticmethod
    def _fmt_time(seconds):
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EscribiendoGUI - Transcriptor de Audio")
        self.resize(700, 500)
        self.setMinimumSize(560, 400)

        self.audio_path = None
        self.worker = None

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(14)

        # Título
        title = QLabel("🎙️ Transcriptor de Audio a Texto")
        title_font = QFont("Segoe UI", 18, QFont.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Botones superiores
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self.btn_select = QPushButton("📂  Seleccionar audio MP3")
        self.btn_select.setCursor(Qt.PointingHandCursor)
        self.btn_select.clicked.connect(self.select_audio)
        btn_row.addWidget(self.btn_select)

        self.btn_annotate = QPushButton("📝  Anotar")
        self.btn_annotate.setCursor(Qt.PointingHandCursor)
        self.btn_annotate.setEnabled(False)
        self.btn_annotate.clicked.connect(self.annotate)
        btn_row.addWidget(self.btn_annotate)

        layout.addLayout(btn_row)

        # Archivo seleccionado
        self.lbl_file = QLabel("Ningún archivo seleccionado")
        self.lbl_file.setAlignment(Qt.AlignCenter)
        self.lbl_file.setStyleSheet("color: #7f8c8d; font-style: italic;")
        layout.addWidget(self.lbl_file)

        # Barra de progreso (visible solo mientras transcribe)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Caja de texto con el resultado (se va llenando en vivo)
        self.result_box = QTextEdit()
        self.result_box.setPlaceholderText(
            "El texto transcrito aparecerá aquí..."
        )
        self.result_box.setFont(QFont("Segoe UI", 11))
        self.result_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.result_box)

        # Barra de estado
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Listo")

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e2630;
            }
            QLabel {
                color: #ecf0f1;
            }
            QPushButton {
                background-color: #2980b9;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 18px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #4a5a6a;
                color: #9aa5b1;
            }
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #3d556e;
                border-radius: 8px;
                padding: 10px;
            }
            QProgressBar {
                background-color: #2c3e50;
                border: 1px solid #3d556e;
                border-radius: 8px;
                color: #ecf0f1;
                text-align: center;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                border-radius: 7px;
            }
            QStatusBar {
                color: #95a5a6;
            }
        """)

    def select_audio(self):
        start_dir = os.path.dirname(os.path.abspath(__file__))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo de audio",
            start_dir,
            "Archivos de audio (*.mp3 *.wav *.m4a *.ogg *.flac);;Todos (*)",
        )
        if path:
            self.audio_path = path
            self.lbl_file.setText(f"📄 {os.path.basename(path)}")
            self.lbl_file.setStyleSheet("color: #2ecc71;")
            self.btn_annotate.setEnabled(True)
            self.status.showMessage("Archivo listo para transcribir")

    def annotate(self):
        if not self.audio_path:
            return
        self.btn_select.setEnabled(False)
        self.btn_annotate.setEnabled(False)
        self.result_box.setPlaceholderText("")
        self.result_box.setPlainText("")

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.worker = TranscriptionWorker(self.audio_path, self)
        self.worker.status.connect(self.status.showMessage)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.chunk_text.connect(self.append_chunk)
        self.worker.finished_ok.connect(self.on_result)
        self.worker.failed.connect(self.on_error)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def append_chunk(self, text):
        """Añade el texto de cada trozo en tiempo real para que el usuario
        vea que la transcripción avanza (nada de pantalla 'congelada')."""
        current = self.result_box.toPlainText()
        sep = " " if current else ""
        self.result_box.setPlainText(current + sep + text)
        self.result_box.moveCursor(QTextCursor.End)

    def closeEvent(self, event):
        # No destruir el QThread mientras sigue transcribiendo;
        # de lo contrario Qt aborta con "QThread: Destroyed while still running".
        if self.worker is not None and self.worker.isRunning():
            self.status.showMessage("Esperando a que termine la transcripción…")
            self.worker.wait()
        super().closeEvent(event)

    def _reset_ui(self):
        self.btn_select.setEnabled(True)
        self.btn_annotate.setEnabled(True)
        self.progress_bar.setVisible(False)

    def on_result(self, text):
        self.progress_bar.setValue(100)
        self.result_box.setPlainText(text)
        self.status.showMessage("Transcripción completada ✔")
        self._reset_ui()

    def on_error(self, message):
        self.result_box.setPlainText(f"Error durante la transcripción:\n{message}")
        self.status.showMessage("Error en la transcripción ✖")
        self._reset_ui()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
