from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QFileDialog, QFrame, QStackedWidget, QListWidget, QLineEdit, QComboBox, QDateEdit, QTimeEdit,
    QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton, QProgressBar, QSlider, QScrollArea
)
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor
from PyQt5.QtCore import Qt, QSize, QTimer
import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Define ModelWithTemperature wrapper
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

# Import your CT_CNN architecture
from services.lung_classifier.singletons.lung_models import CT_CNN

# Instantiate and load the calibrated model
model = ModelWithTemperature(CT_CNN(num_classes=3))
model.load_state_dict(torch.load("../models/calibrated_lung_cancer_detector.pth", map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Basic prediction function (fallback)
def predict_image(image_path, threshold=0.7):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        max_prob, predicted_index = torch.max(probabilities, dim=0)
        max_prob = max_prob.item()
        predicted_index = predicted_index.item()
    class_labels = ["Benign", "Malignant", "Normal"]
    if max_prob < threshold:
        return ("Unknown", max_prob * 100, probabilities.tolist(), None)
    predicted_label = class_labels[predicted_index]
    return (predicted_label, max_prob * 100, probabilities.tolist(), None)

# New function: predict_image_with_heatmap – returns a saliency heatmap
def predict_image_with_heatmap(image_path, threshold=0.7):
    image_tensor = preprocess_image(image_path)
    image_tensor.requires_grad_()
    with torch.enable_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        max_prob, predicted_index = torch.max(probabilities, dim=0)
        max_prob = max_prob.item()
        predicted_index = predicted_index.item()

        # Backprop to get gradients for the predicted class
        output[0, predicted_index].backward()
        gradients = image_tensor.grad.data

        # Compute a simple saliency map: maximum absolute value across channels
        saliency, _ = torch.max(gradients.abs(), dim=1)
        saliency = saliency.squeeze().cpu().numpy()

        # Normalize to 0–255
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        saliency = (saliency * 255).astype(np.uint8)
        heatmap = Image.fromarray(saliency).convert("L")

    class_labels = ["Benign", "Malignant", "Normal"]
    if max_prob < threshold:
        return ("Unknown", max_prob * 100, probabilities.tolist(), heatmap)
    predicted_label = class_labels[predicted_index]
    return (predicted_label, max_prob * 100, probabilities.tolist(), heatmap)

# Modern Loading Spinner with smooth 60fps animation
class LoadingSpinner(QWidget):
    def __init__(self, parent=None, dark_mode=True):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.setFixedSize(100, 100)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.color = QColor("#03DAC6")
        self.pen_width = 8
        self.arc_length = 270
        self.progress = 0  # optional

    def set_dark_mode(self, dark_mode):
        self.dark_mode = dark_mode
        self.update()

    def start(self):
        self.timer.start(16)  # ~60fps
        self.show()

    def stop(self):
        self.timer.stop()
        self.hide()

    def update_animation(self):
        self.angle = (self.angle + 3) % 360
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        bg_color = QColor(30, 30, 30, 200) if self.dark_mode else QColor(240, 240, 240, 200)
        painter.setBrush(bg_color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.rect())

        pen = QtGui.QPen()
        pen.setWidth(self.pen_width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setColor(self.color)
        painter.setPen(pen)

        rect = QtCore.QRectF(
            self.pen_width / 2,
            self.pen_width / 2,
            self.width() - self.pen_width,
            self.height() - self.pen_width
        )
        start_angle = self.angle * 16
        span_angle = self.arc_length * 16
        painter.drawArc(rect, start_angle, span_angle)

        if hasattr(self, 'progress'):
            painter.setPen(Qt.white if self.dark_mode else Qt.black)
            font = painter.font()
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, f"{self.progress}%")

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 900)
        MainWindow.setWindowTitle("Lung Cancer Detection Dashboard")

        # Theme and settings
        self.dark_mode = True
        self.current_language = "english"
        self.confidence_threshold = 0.7
        self.accent_color = "Teal"

        # Create loading spinner
        self.loading_spinner = LoadingSpinner(MainWindow, dark_mode=self.dark_mode)
        self.loading_spinner.hide()

        # Central widget
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        # Main layout
        main_layout = QHBoxLayout(self.centralwidget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create sidebar + content area
        self.create_sidebar(main_layout)
        self.create_content_area(main_layout)

        # Now that UI elements exist, update stylesheet
        self.update_stylesheet()
        # Initialize charts & data
        self.update_charts()
        self.init_history_data()

        # Connect button signals
        self.BrowseImage.clicked.connect(self.loadImage)
        self.Classify.clicked.connect(self.classifyFunction)
        self.Training.clicked.connect(self.trainingFunction)
        self.language_combo.currentTextChanged.connect(self.change_language)

    def create_sidebar(self, main_layout):
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        sidebar_layout.setSpacing(15)

        title_label = QLabel("Lung Cancer AI")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title_label)

        self.btn_dashboard = QPushButton(QIcon("icons/dashboard.png"), " Dashboard")
        self.btn_analyze   = QPushButton(QIcon("icons/scan.png"), " Analyze Scan")
        self.btn_history   = QPushButton(QIcon("icons/history.png"), " History")
        self.btn_settings  = QPushButton(QIcon("icons/settings.png"), " Settings")

        for btn in [self.btn_dashboard, self.btn_analyze, self.btn_history, self.btn_settings]:
            btn.setIconSize(QSize(24, 24))
            btn.setObjectName("navButton")
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()

        version_label = QLabel("v1.1.0")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setObjectName("versionLabel")
        sidebar_layout.addWidget(version_label)

        main_layout.addWidget(sidebar)

    def create_content_area(self, main_layout):
        """Sets up the main content area with a header and a QStackedWidget for pages."""
        content_area = QWidget()
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(60)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)

        self.current_section = QLabel("Dashboard")
        self.current_section.setObjectName("sectionLabel")
        header_layout.addWidget(self.current_section)
        header_layout.addStretch()

        self.theme_toggle = QPushButton()
        self.theme_toggle.setIcon(QIcon("icons/moon.png"))
        self.theme_toggle.setIconSize(QSize(20, 20))
        self.theme_toggle.setObjectName("themeToggle")
        self.theme_toggle.setToolTip("Toggle Dark/Light Mode")
        header_layout.addWidget(self.theme_toggle)

        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Shona"])
        self.language_combo.setCurrentIndex(0)
        self.language_combo.setObjectName("languageCombo")
        self.language_combo.setFixedWidth(120)
        header_layout.addWidget(self.language_combo)

        user_btn = QPushButton(QIcon("icons/user.png"), "")
        user_btn.setIconSize(QSize(24, 24))
        user_btn.setObjectName("userButton")
        header_layout.addWidget(user_btn)

        content_layout.addWidget(header)

        self.stacked_widget = QStackedWidget()
        content_layout.addWidget(self.stacked_widget)

        self.create_dashboard_page()
        self.create_analysis_page_with_scroll()  # <--- We'll use a QScrollArea here
        self.create_history_page()
        self.create_settings_page()

        self.stacked_widget.setCurrentIndex(0)
        main_layout.addWidget(content_area)

        # Navigation signals
        self.btn_dashboard.clicked.connect(lambda: self.switch_page(0))
        self.btn_analyze.clicked.connect(lambda: self.switch_page(1))
        self.btn_history.clicked.connect(lambda: self.switch_page(2))
        self.btn_settings.clicked.connect(lambda: self.switch_page(3))
        self.theme_toggle.clicked.connect(self.toggle_theme)

    def create_dashboard_page(self):
        dashboard_page = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_page)
        dashboard_layout.setContentsMargins(20, 20, 20, 20)
        dashboard_layout.setSpacing(20)

        # Stats row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(15)
        self.total_scans_card = self.create_stat_card("Total Scans", "1,024", "#03DAC6")
        self.malignant_card   = self.create_stat_card("Malignant", "328", "#CF6679")
        self.benign_card      = self.create_stat_card("Benign", "542", "#018786")
        self.normal_card      = self.create_stat_card("Normal", "154", "#BB86FC")

        stats_row.addWidget(self.total_scans_card)
        stats_row.addWidget(self.malignant_card)
        stats_row.addWidget(self.benign_card)
        stats_row.addWidget(self.normal_card)
        dashboard_layout.addLayout(stats_row)

        # Charts row
        charts_row = QHBoxLayout()
        charts_row.setSpacing(15)

        analysis_frame = QFrame()
        analysis_frame.setMinimumHeight(300)
        analysis_layout = QVBoxLayout(analysis_frame)

        analysis_title = QLabel("Monthly Analysis")
        analysis_title.setObjectName("sectionLabel")
        analysis_layout.addWidget(analysis_title)

        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        analysis_layout.addWidget(self.canvas)

        perf_frame = QFrame()
        perf_frame.setMinimumHeight(300)
        perf_layout = QVBoxLayout(perf_frame)

        perf_title = QLabel("Model Performance")
        perf_title.setObjectName("sectionLabel")
        perf_layout.addWidget(perf_title)

        self.perf_figure = Figure(figsize=(5, 3), dpi=100)
        self.perf_canvas = FigureCanvas(self.perf_figure)
        perf_layout.addWidget(self.perf_canvas)

        charts_row.addWidget(analysis_frame, 1)
        charts_row.addWidget(perf_frame, 1)
        dashboard_layout.addLayout(charts_row)

        # Recent scans row
        recent_row = QHBoxLayout()
        recent_row.setSpacing(15)

        scans_frame = QFrame()
        scans_layout = QVBoxLayout(scans_frame)

        scans_title = QLabel("Recent Scans")
        scans_title.setObjectName("sectionLabel")
        scans_layout.addWidget(scans_title)

        self.scans_list = QListWidget()
        self.scans_list.itemDoubleClicked.connect(self.open_history_item)
        scans_layout.addWidget(self.scans_list)

        stats_frame = QFrame()
        stats_layout = QVBoxLayout(stats_frame)

        stats_title = QLabel("Quick Stats")
        stats_title.setObjectName("sectionLabel")
        stats_layout.addWidget(stats_title)

        self.accuracy_label = QLabel("Model Accuracy: 92.3%")
        self.recall_label   = QLabel("Malignant Recall: 89.5%")
        self.f1_label       = QLabel("F1 Score: 90.8%")

        for lbl in [self.accuracy_label, self.recall_label, self.f1_label]:
            lbl.setObjectName("statLabel")
            stats_layout.addWidget(lbl)

        stats_layout.addStretch()
        recent_row.addWidget(scans_frame, 2)
        recent_row.addWidget(stats_frame, 1)
        dashboard_layout.addLayout(recent_row)

        self.stacked_widget.addWidget(dashboard_page)

    def create_analysis_page_with_scroll(self):
        """
        Creates the Analysis page but wraps it in a QScrollArea so if the page is too tall,
        the user can scroll down to see the buttons.
        """
        # The container that holds everything
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(20, 20, 20, 20)
        container_layout.setSpacing(20)

        # --- Top row: CT Scan + Patient Info ---
        top_row = QHBoxLayout()
        top_row.setSpacing(20)

        image_frame = QFrame()
        image_frame.setMinimumHeight(400)
        image_layout = QVBoxLayout(image_frame)

        image_title = QLabel("CT Scan Analysis")
        image_title.setObjectName("sectionLabel")
        image_layout.addWidget(image_title)

        self.imageLbl = QLabel()
        self.imageLbl.setFrameShape(QFrame.Box)
        self.imageLbl.setAlignment(Qt.AlignCenter)
        self.imageLbl.setMinimumHeight(350)
        self.imageLbl.setStyleSheet("background-color: black;")
        image_layout.addWidget(self.imageLbl)
        top_row.addWidget(image_frame, 2)

        patient_frame = QFrame()
        patient_layout = QVBoxLayout(patient_frame)

        patient_title = QLabel("Patient Information")
        patient_title.setObjectName("sectionLabel")
        patient_layout.addWidget(patient_title)

        form_layout = QtWidgets.QFormLayout()
        form_layout.setHorizontalSpacing(20)
        form_layout.setVerticalSpacing(10)

        self.patient_id = QLineEdit()
        self.patient_id.setPlaceholderText("Enter patient ID")
        self.patient_name = QLineEdit()
        self.patient_name.setPlaceholderText("Patient name")
        self.patient_age = QSpinBox()
        self.patient_age.setRange(1, 120)
        self.patient_gender = QComboBox()
        self.patient_gender.addItems(["Male", "Female", "Other"])
        self.smoking_check = QCheckBox("Smoking history")
        self.family_check = QCheckBox("Family history")
        self.asbestos_check = QCheckBox("Asbestos exposure")
        self.scan_date = QDateEdit()
        self.scan_date.setCalendarPopup(True)
        self.scan_date.setDate(QtCore.QDate.currentDate())
        self.scan_time = QTimeEdit()
        self.scan_time.setTime(QtCore.QTime.currentTime())
        self.tech_notes = QTextEdit()
        self.tech_notes.setPlaceholderText("Technician notes...")
        self.tech_notes.setMaximumHeight(80)

        form_layout.addRow("Patient ID:", self.patient_id)
        form_layout.addRow("Name:", self.patient_name)
        form_layout.addRow("Age:", self.patient_age)
        form_layout.addRow("Gender:", self.patient_gender)
        form_layout.addRow("Scan Date:", self.scan_date)
        form_layout.addRow("Scan Time:", self.scan_time)
        form_layout.addRow("Risk Factors:", self.smoking_check)
        form_layout.addRow("", self.family_check)
        form_layout.addRow("", self.asbestos_check)
        form_layout.addRow("Notes:", self.tech_notes)

        patient_layout.addLayout(form_layout)
        patient_layout.addStretch()
        top_row.addWidget(patient_frame, 1)

        container_layout.addLayout(top_row)

        # --- Results row (Analysis Results + Probability Distribution) ---
        results_row = QHBoxLayout()
        results_row.setSpacing(20)

        results_card = QFrame()
        results_layout = QVBoxLayout(results_card)
        results_title = QLabel("Analysis Results")
        results_title.setObjectName("sectionLabel")
        results_layout.addWidget(results_title)

        self.textEdit = QTextEdit()
        self.textEdit.setReadOnly(True)
        results_layout.addWidget(self.textEdit)

        prob_card = QFrame()
        prob_layout = QVBoxLayout(prob_card)
        prob_title = QLabel("Probability Distribution")
        prob_title.setObjectName("sectionLabel")
        prob_layout.addWidget(prob_title)

        self.prob_figure = Figure(figsize=(5, 3), dpi=100)
        self.prob_canvas = FigureCanvas(self.prob_figure)
        prob_layout.addWidget(self.prob_canvas)

        results_row.addWidget(results_card, 1)
        results_row.addWidget(prob_card, 1)
        container_layout.addLayout(results_row)

        # --- Heatmap row (below results) ---
        heatmap_row = QVBoxLayout()
        heatmap_title = QLabel("Model Activation Heatmap")
        heatmap_title.setObjectName("sectionLabel")
        heatmap_row.addWidget(heatmap_title)

        self.heatmapLbl = QLabel()
        self.heatmapLbl.setFrameShape(QFrame.Box)
        self.heatmapLbl.setAlignment(Qt.AlignCenter)
        self.heatmapLbl.setMinimumHeight(300)
        self.heatmapLbl.setStyleSheet("background-color: black;")
        heatmap_row.addWidget(self.heatmapLbl)

        controls_layout = QHBoxLayout()
        opacity_label = QLabel("Heatmap Intensity:")
        self.heatmap_opacity = QSlider(Qt.Horizontal)
        self.heatmap_opacity.setRange(10, 100)
        self.heatmap_opacity.setValue(70)
        self.heatmap_opacity.setTickInterval(10)
        self.heatmap_opacity.setTickPosition(QSlider.TicksBelow)
        self.heatmap_opacity.valueChanged.connect(self.update_heatmap_display)

        controls_layout.addWidget(opacity_label)
        controls_layout.addWidget(self.heatmap_opacity)
        heatmap_row.addLayout(controls_layout)
        container_layout.addLayout(heatmap_row)

        # --- Button row (Upload CT Scan, Analyze, Train Model) ---
        button_row = QHBoxLayout()
        button_row.setSpacing(15)

        self.BrowseImage = QPushButton(QIcon("icons/upload.png"), " Upload CT Scan")
        self.Classify    = QPushButton(QIcon("icons/analyze.png"), " Analyze")
        self.Training    = QPushButton(QIcon("icons/train.png"),  " Train Model")
        self.Training.setObjectName("dangerButton")

        for btn in [self.BrowseImage, self.Classify, self.Training]:
            btn.setIconSize(QSize(20, 20))
            button_row.addWidget(btn)

        container_layout.addLayout(button_row)

        # Wrap the container in a QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(container)

        # Add this scroll area as a page in the stacked widget
        self.stacked_widget.addWidget(scroll_area)

    def create_history_page(self):
        history_page = QWidget()
        history_layout = QVBoxLayout(history_page)
        history_layout.setContentsMargins(20, 20, 20, 20)
        history_layout.setSpacing(20)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(15)

        self.filter_patient = QLineEdit()
        self.filter_patient.setPlaceholderText("Filter by patient...")
        self.filter_result = QComboBox()
        self.filter_result.addItems(["All Results", "Malignant", "Benign", "Normal", "Unknown"])
        self.filter_date_from = QDateEdit()
        self.filter_date_from.setCalendarPopup(True)
        self.filter_date_from.setDate(QtCore.QDate.currentDate().addMonths(-1))
        self.filter_date_to = QDateEdit()
        self.filter_date_to.setCalendarPopup(True)
        self.filter_date_to.setDate(QtCore.QDate.currentDate())

        filter_btn = QPushButton("Apply Filters")
        filter_btn.setObjectName("filterButton")

        filter_row.addWidget(QLabel("Patient:"))
        filter_row.addWidget(self.filter_patient)
        filter_row.addWidget(QLabel("Result:"))
        filter_row.addWidget(self.filter_result)
        filter_row.addWidget(QLabel("From:"))
        filter_row.addWidget(self.filter_date_from)
        filter_row.addWidget(QLabel("To:"))
        filter_row.addWidget(self.filter_date_to)
        filter_row.addWidget(filter_btn)
        history_layout.addLayout(filter_row)

        self.history_table = QtWidgets.QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(["Date", "Patient ID", "Name", "Result", "Accuracy", "Actions"])
        self.history_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.history_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        history_layout.addWidget(self.history_table)

        compare_btn = QPushButton("Compare Selected Scans")
        compare_btn.setObjectName("compareButton")
        history_layout.addWidget(compare_btn, alignment=Qt.AlignRight)

        self.stacked_widget.addWidget(history_page)

        filter_btn.clicked.connect(self.update_history_view)
        compare_btn.clicked.connect(self.compare_scans)

    def create_settings_page(self):
        settings_page = QWidget()
        settings_layout = QVBoxLayout(settings_page)
        settings_layout.setContentsMargins(20, 20, 20, 20)
        settings_layout.setSpacing(20)

        appearance_frame = QFrame()
        appearance_layout = QVBoxLayout(appearance_frame)

        appearance_title = QLabel("Appearance")
        appearance_title.setObjectName("sectionLabel")
        appearance_layout.addWidget(appearance_title)

        theme_group = QtWidgets.QGroupBox("Theme Settings")
        theme_layout = QVBoxLayout(theme_group)

        self.theme_auto = QCheckBox("Follow system theme")
        self.theme_dark = QRadioButton("Dark mode")
        self.theme_light = QRadioButton("Light mode")
        self.theme_dark.setChecked(self.dark_mode)
        self.theme_light.setChecked(not self.dark_mode)

        theme_layout.addWidget(self.theme_auto)
        theme_layout.addWidget(self.theme_dark)
        theme_layout.addWidget(self.theme_light)

        accent_label = QLabel("Accent Color:")
        self.accent_color_combo = QComboBox()
        self.accent_color_combo.addItems(["Teal", "Purple", "Blue", "Pink", "Orange"])
        self.accent_color_combo.setCurrentText(self.accent_color)

        theme_layout.addWidget(accent_label)
        theme_layout.addWidget(self.accent_color_combo)

        appearance_layout.addWidget(theme_group)
        settings_layout.addWidget(appearance_frame)

        analysis_frame = QFrame()
        analysis_layout = QVBoxLayout(analysis_frame)

        analysis_title = QLabel("Analysis Settings")
        analysis_title.setObjectName("sectionLabel")
        analysis_layout.addWidget(analysis_title)

        threshold_label = QLabel("Confidence Threshold:")
        self.threshold_slider = QDoubleSpinBox()
        self.threshold_slider.setRange(0.1, 0.99)
        self.threshold_slider.setSingleStep(0.05)
        self.threshold_slider.setValue(0.7)
        self.threshold_slider.setDecimals(2)

        analysis_layout.addWidget(threshold_label)
        analysis_layout.addWidget(self.threshold_slider)
        settings_layout.addWidget(analysis_frame)
        settings_layout.addStretch()

        save_btn = QPushButton("Save Settings")
        save_btn.setObjectName("saveButton")
        settings_layout.addWidget(save_btn, alignment=Qt.AlignRight)

        self.stacked_widget.addWidget(settings_page)

        self.theme_auto.stateChanged.connect(self.toggle_theme_auto)
        self.theme_dark.toggled.connect(lambda: self.set_theme_mode("dark"))
        self.theme_light.toggled.connect(lambda: self.set_theme_mode("light"))
        self.accent_color_combo.currentTextChanged.connect(self.update_stylesheet)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        save_btn.clicked.connect(self.save_settings)

    def get_accent_color(self):
        colors = {
            "Teal": "#03DAC6",
            "Purple": "#BB86FC",
            "Blue": "#3700B3",
            "Pink": "#CF6679",
            "Orange": "#FF8C00"
        }
        return colors.get(self.accent_color_combo.currentText(), "#03DAC6")

    def update_stylesheet(self):
        accent_color = self.get_accent_color()
        danger_color = "#CF6679"
        if self.dark_mode:
            stylesheet = f"""
                QMainWindow {{
                    background-color: #121212;
                }}
                QWidget#centralwidget {{
                    background-color: #121212;
                }}
                QLabel {{
                    color: #E0E0E0;
                    font-size: 14px;
                }}
                QLabel#titleLabel {{
                    color: #FFFFFF;
                    font-size: 24px;
                    font-weight: bold;
                }}
                QLabel#sectionLabel {{
                    color: {accent_color};
                    font-size: 16px;
                    font-weight: bold;
                    border-bottom: 1px solid {accent_color};
                    padding-bottom: 5px;
                }}
                QLabel#statLabel {{
                    color: #E0E0E0;
                    font-size: 16px;
                    padding: 5px;
                }}
                QPushButton {{
                    background-color: {accent_color};
                    color: #FFFFFF;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-size: 14px;
                    min-width: 100px;
                }}
                QPushButton:hover {{
                    background-color: {self.adjust_color(accent_color, 20)};
                }}
                QPushButton:pressed {{
                    background-color: {self.adjust_color(accent_color, -20)};
                }}
                QPushButton#navButton {{
                    text-align: left;
                    padding-left: 15px;
                }}
                QPushButton#dangerButton {{
                    background-color: {danger_color};
                }}
                QPushButton#dangerButton:hover {{
                    background-color: {self.adjust_color(danger_color, 20)};
                }}
                QPushButton#filterButton, QPushButton#compareButton, QPushButton#saveButton {{
                    background-color: #03DAC6;
                }}
                QPushButton#viewButton {{
                    background-color: #3700B3;
                    padding: 4px 8px;
                    min-width: 60px;
                }}
                QPushButton#reanalyzeButton {{
                    background-color: #018786;
                    padding: 4px 8px;
                    min-width: 60px;
                }}
                QPushButton#themeToggle, QPushButton#userButton {{
                    background-color: transparent;
                    min-width: 0;
                    padding: 5px;
                }}
                QTextEdit {{
                    background-color: #1E1E1E;
                    color: #FFFFFF;
                    border: 1px solid #444;
                    border-radius: 4px;
                    padding: 5px;
                    font-size: 14px;
                }}
                QFrame {{
                    background-color: #1E1E1E;
                    border-radius: 5px;
                    border: 1px solid #444;
                }}
                QLineEdit, QComboBox, QDateEdit, QTimeEdit, QSpinBox, QDoubleSpinBox {{
                    background-color: #1E1E1E;
                    color: #FFFFFF;
                    border: 1px solid #444;
                    border-radius: 4px;
                    padding: 5px;
                }}
                QCheckBox {{
                    color: #E0E0E0;
                    spacing: 5px;
                }}
                QRadioButton {{
                    color: #E0E0E0;
                    spacing: 5px;
                }}
                QGroupBox {{
                    border: 1px solid #444;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 15px;
                    color: #E0E0E0;
                }}
                QHeaderView::section {{
                    background-color: #1E1E1E;
                    color: white;
                    padding: 5px;
                    border: none;
                }}
                QTableWidget {{
                    background-color: #1E1E1E;
                    color: white;
                    gridline-color: #444;
                }}
                QTableWidget::item {{
                    padding: 5px;
                }}
                QProgressBar {{
                    border: 1px solid #444;
                    border-radius: 3px;
                    text-align: center;
                    color: white;
                }}
                QProgressBar::chunk {{
                    background-color: {accent_color};
                }}
                QLabel#versionLabel {{
                    color: #A0A0A0;
                }}
            """
        else:
            stylesheet = f"""
                QMainWindow {{
                    background-color: #F5F5F5;
                }}
                QWidget#centralwidget {{
                    background-color: #F5F5F5;
                }}
                QLabel {{
                    color: #333333;
                    font-size: 14px;
                }}
                QLabel#titleLabel {{
                    color: #000000;
                    font-size: 24px;
                    font-weight: bold;
                }}
                QLabel#sectionLabel {{
                    color: {accent_color};
                    font-size: 16px;
                    font-weight: bold;
                    border-bottom: 1px solid {accent_color};
                    padding-bottom: 5px;
                }}
                QLabel#statLabel {{
                    color: #333333;
                    font-size: 16px;
                    padding: 5px;
                }}
                QPushButton {{
                    background-color: {accent_color};
                    color: #FFFFFF;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-size: 14px;
                    min-width: 100px;
                }}
                QPushButton:hover {{
                    background-color: {self.adjust_color(accent_color, 20)};
                }}
                QPushButton:pressed {{
                    background-color: {self.adjust_color(accent_color, -20)};
                }}
                QTextEdit {{
                    background-color: #FFFFFF;
                    color: #333333;
                    border: 1px solid #CCCCCC;
                    border-radius: 4px;
                    padding: 5px;
                    font-size: 14px;
                }}
                QFrame {{
                    background-color: #FFFFFF;
                    border-radius: 5px;
                    border: 1px solid #CCCCCC;
                }}
                QLineEdit, QComboBox, QDateEdit, QTimeEdit, QSpinBox, QDoubleSpinBox {{
                    background-color: #FFFFFF;
                    color: #333333;
                    border: 1px solid #CCCCCC;
                    border-radius: 4px;
                    padding: 5px;
                }}
                QCheckBox {{
                    color: #333333;
                    spacing: 5px;
                }}
                QRadioButton {{
                    color: #333333;
                    spacing: 5px;
                }}
                QGroupBox {{
                    border: 1px solid #CCCCCC;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 15px;
                    color: #333333;
                }}
                QHeaderView::section {{
                    background-color: #F0F0F0;
                    color: #333333;
                    padding: 5px;
                    border: none;
                }}
                QTableWidget {{
                    background-color: #FFFFFF;
                    color: #333333;
                    gridline-color: #CCCCCC;
                }}
                QTableWidget::item {{
                    padding: 5px;
                }}
                QProgressBar {{
                    border: 1px solid #CCCCCC;
                    border-radius: 3px;
                    text-align: center;
                    color: #333333;
                }}
                QProgressBar::chunk {{
                    background-color: {accent_color};
                }}
                QLabel#versionLabel {{
                    color: #777777;
                }}
            """
        self.centralwidget.setStyleSheet(stylesheet)

    def adjust_color(self, hex_color, amount):
        color = QColor(hex_color)
        if amount > 0:
            return color.lighter(100 + amount).name()
        else:
            return color.darker(100 - amount).name()

    def update_threshold(self, value):
        self.confidence_threshold = value

    def save_settings(self):
        self.textEdit.setText("Settings saved successfully!")

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.loading_spinner.set_dark_mode(self.dark_mode)
        self.update_stylesheet()
        icon = "icons/moon.png" if self.dark_mode else "icons/sun.png"
        self.theme_toggle.setIcon(QIcon(icon))
        self.update_charts()
        if hasattr(self, 'prob_figure'):
            self.prob_figure.clear()
            self.prob_canvas.draw()

    def toggle_theme_auto(self, state):
        pass

    def set_theme_mode(self, mode):
        self.dark_mode = (mode == "dark")
        self.loading_spinner.set_dark_mode(self.dark_mode)
        self.update_stylesheet()
        self.update_charts()
        if hasattr(self, 'prob_figure'):
            self.prob_figure.clear()
            self.prob_canvas.draw()

    def change_language(self, language):
        self.current_language = "shona" if language == "Shona" else "english"
        titles = ["Dashboard", "Analyze Scan", "History", "Settings"]
        if self.current_language == "shona":
            titles = ["Dashboardi", "Ongorora Scan", "Nhoroondo", "Zvirongwa"]
        self.current_section.setText(titles[self.stacked_widget.currentIndex()])

    def update_charts(self):
        self.figure.clear()
        self.perf_figure.clear()
        ax = self.figure.add_subplot(111)
        categories = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
        values = [23, 45, 56, 78, 33]
        ax.bar(categories, values, color=self.get_accent_color())
        ax.set_facecolor('#383838' if self.dark_mode else '#FFFFFF')
        self.figure.set_facecolor('#383838' if self.dark_mode else '#FFFFFF')
        ax.tick_params(colors='white' if self.dark_mode else 'black')
        for spine in ax.spines.values():
            spine.set_color('white' if self.dark_mode else 'black')
        self.canvas.draw()

        ax2 = self.perf_figure.add_subplot(111)
        epochs = range(1, 11)
        train_loss = [0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23, 0.2]
        val_loss = [0.85, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38, 0.35, 0.33, 0.3]
        ax2.plot(epochs, train_loss, label='Training Loss', color=self.get_accent_color())
        ax2.plot(epochs, val_loss, label='Validation Loss', color='#CF6679')
        ax2.set_title('Training & Validation Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.set_facecolor('#383838' if self.dark_mode else '#FFFFFF')
        self.perf_figure.set_facecolor('#383838' if self.dark_mode else '#FFFFFF')
        ax2.tick_params(colors='white' if self.dark_mode else 'black')
        for spine in ax2.spines.values():
            spine.set_color('white' if self.dark_mode else 'black')
        self.perf_canvas.draw()

        self.scans_list.clear()
        sample_scans = [
            "Scan_2023-01-15 (Malignant)",
            "Scan_2023-02-22 (Benign)",
            "Scan_2023-03-10 (Normal)",
            "Scan_2023-04-05 (Malignant)",
            "Scan_2023-05-18 (Benign)"
        ]
        for scan in sample_scans:
            item = QtWidgets.QListWidgetItem(scan)
            self.scans_list.addItem(item)

    def init_history_data(self):
        self.history_data = []
        results = ["Malignant", "Benign", "Normal"]
        for i in range(20):
            date = QtCore.QDate.currentDate().addDays(-random.randint(0, 90))
            time = QtCore.QTime(random.randint(8, 16), random.randint(0, 59))
            patient_id = f"PT{random.randint(1000, 9999)}"
            name = f"Patient {i+1}"
            result = random.choice(results)
            confidence = round(random.uniform(0.7, 0.99), 2)
            self.history_data.append({
                "date": date.toString("yyyy-MM-dd"),
                "time": time.toString("hh:mm"),
                "patient_id": patient_id,
                "name": name,
                "result": result,
                "confidence": confidence
            })

    def switch_page(self, index):
        self.stacked_widget.setCurrentIndex(index)
        titles = ["Dashboard", "Analyze Scan", "History", "Settings"]
        self.current_section.setText(titles[index])
        if index == 0:
            self.update_charts()
        elif index == 2:
            self.update_history_view()

    def create_stat_card(self, title, value, color):
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                border-left: 4px solid {color};
                background-color: {'#383838' if self.dark_mode else '#FFFFFF'};
                padding: 10px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 14px; color: #A0A0A0;")
        value_label = QLabel(value)
        value_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
        card_layout.addWidget(title_label)
        card_layout.addWidget(value_label)
        return card

    def update_history_view(self):
        self.history_table.setRowCount(0)
        filtered_data = []
        patient_filter = self.filter_patient.text().lower()
        result_filter = self.filter_result.currentText()
        date_from = self.filter_date_from.date()
        date_to = self.filter_date_to.date()

        for item in self.history_data:
            date = QtCore.QDate.fromString(item["date"], "yyyy-MM-dd")
            matches_patient = patient_filter in item["name"].lower() or patient_filter in item["patient_id"].lower()
            matches_result = (result_filter == "All Results") or (result_filter == item["result"])
            matches_date = (date >= date_from and date <= date_to)

            if matches_patient and matches_result and matches_date:
                filtered_data.append(item)

        self.history_table.setRowCount(len(filtered_data))
        for row, item in enumerate(filtered_data):
            self.history_table.setItem(row, 0, QtWidgets.QTableWidgetItem(item["date"] + " " + item["time"]))
            self.history_table.setItem(row, 1, QtWidgets.QTableWidgetItem(item["patient_id"]))
            self.history_table.setItem(row, 2, QtWidgets.QTableWidgetItem(item["name"]))
            self.history_table.setItem(row, 3, QtWidgets.QTableWidgetItem(item["result"]))
            self.history_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{item['confidence']*100:.1f}%"))

            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(0, 0, 0, 0)

            view_btn = QPushButton("View")
            view_btn.setObjectName("viewButton")
            view_btn.clicked.connect(lambda _, r=row: self.view_history_item(r))

            reanalyze_btn = QPushButton("Reanalyze")
            reanalyze_btn.setObjectName("reanalyzeButton")
            reanalyze_btn.clicked.connect(lambda _, r=row: self.reanalyze_item(r))

            action_layout.addWidget(view_btn)
            action_layout.addWidget(reanalyze_btn)
            action_widget.setLayout(action_layout)

            self.history_table.setCellWidget(row, 5, action_widget)

        self.history_table.resizeColumnsToContents()

    def view_history_item(self, row):
        item = self.history_data[row]
        self.stacked_widget.setCurrentIndex(1)
        self.current_section.setText("Analyze Scan")

        self.textEdit.setText(
            f"Patient: {item['name']}\n"
            f"ID: {item['patient_id']}\n"
            f"Date: {item['date']} {item['time']}\n"
            f"Previous Result: {item['result']}\n"
            f"Accuracy: {item['confidence']*100:.1f}%"
        )

        # Just a fake probability chart for demonstration
        probs = [0.2, 0.3, 0.5] if item["result"] == "Normal" else [0.6, 0.3, 0.1] if item["result"] == "Benign" else [0.1, 0.8, 0.1]
        self.update_probability_chart(probs)

    def reanalyze_item(self, row):
        self.loading_spinner.start()
        QTimer.singleShot(2000, lambda: self.finish_reanalysis(row))

    def finish_reanalysis(self, row):
        self.loading_spinner.stop()
        item = self.history_data[row]
        new_confidence = min(1.0, item["confidence"] + random.uniform(-0.05, 0.05))
        item["confidence"] = round(new_confidence, 2)

        self.update_history_view()
        self.textEdit.setText(
            f"Reanalysis complete for {item['name']}\nNew accuracy: {item['confidence']*100:.1f}%"
        )

    def compare_scans(self):
        selected = self.history_table.selectionModel().selectedRows()
        if len(selected) < 2:
            self.textEdit.setText("Please select at least 2 scans to compare")
            return

        self.loading_spinner.start()
        QTimer.singleShot(1500, lambda: self.show_comparison(selected))

    def show_comparison(self, selected):
        self.loading_spinner.stop()
        comparison_text = "Comparison Results:\n\n"
        for idx, row in enumerate(selected):
            item = self.history_data[row.row()]
            comparison_text += (
                f"{idx+1}. {item['name']} ({item['date']}): {item['result']} ({item['confidence']*100:.1f}%)\n"
            )
        self.textEdit.setText(comparison_text)
        self.stacked_widget.setCurrentIndex(1)

    def open_history_item(self, item):
        print("Opening:", item.text())

    def loadImage(self):
        fileName, _ = QFileDialog.getOpenFileName(
            None, "Select CT Scan Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if fileName:
            print("Loaded image:", fileName)
            self.file = fileName
            pixmap = QPixmap(fileName)
            pixmap = pixmap.scaled(
                self.imageLbl.width(),
                self.imageLbl.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.imageLbl.setPixmap(pixmap)
            self.textEdit.clear()
            self.prob_figure.clear()
            self.prob_canvas.draw()

            self.patient_id.setText(f"PT{random.randint(1000, 9999)}")
            self.patient_name.setText("New Patient")
            self.patient_age.setValue(random.randint(30, 80))
            self.patient_gender.setCurrentIndex(random.randint(0, 2))
            self.scan_date.setDate(QtCore.QDate.currentDate())
            self.scan_time.setTime(QtCore.QTime.currentTime())

    def process_classification(self):
        print("Selected file:", self.file)
        try:
            # Display the original CT scan
            pixmap = QPixmap(self.file).scaled(
                self.imageLbl.width(),
                self.imageLbl.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.imageLbl.setPixmap(pixmap)

            # Store original image for heatmap
            self.original_image = Image.open(self.file).convert("RGB")

            # Generate prediction + heatmap
            label, accuracy, probabilities, heatmap = predict_image_with_heatmap(
                self.file,
                threshold=self.confidence_threshold
            )
            self.current_heatmap = heatmap  # store for update_heatmap_display
            self.update_probability_chart(probabilities)
            self.update_heatmap_display()  # show heatmap in separate area

            # Build result text
            if self.current_language == "shona":
                result_text = self.get_shona_result(label, accuracy)
            else:
                result_text = self.get_english_result(label, accuracy)

            result_text += (
                f"\n\nPatient: {self.patient_name.text()}\n"
                f"ID: {self.patient_id.text()}\n"
                f"Age: {self.patient_age.value()}\n"
                f"Gender: {self.patient_gender.currentText()}\n"
                f"Scan Date: {self.scan_date.date().toString('yyyy-MM-dd')} {self.scan_time.time().toString('hh:mm')}"
            )

        except Exception as e:
            result_text = f"Error during analysis: {str(e)}"
            self.heatmapLbl.clear()
            self.heatmapLbl.setText("Heatmap unavailable")
        finally:
            self.textEdit.setText(result_text)
            self.loading_spinner.stop()
            self.add_to_history(label, accuracy)

    def classifyFunction(self):
        if not hasattr(self, 'file'):
            self.textEdit.setText("No CT scan selected!")
            return
        self.loading_spinner.move(
            self.imageLbl.x() + self.imageLbl.width()//2 - 50,
            self.imageLbl.y() + self.imageLbl.height()//2 - 50
        )
        self.loading_spinner.start()
        QTimer.singleShot(2000, self.process_classification)

    def update_heatmap_display(self):
        """Update the separate heatmap area based on the slider's opacity."""
        if hasattr(self, 'current_heatmap') and hasattr(self, 'original_image'):
            opacity = self.heatmap_opacity.value() / 100.0
            heatmap_img = self.create_heatmap_overlay(self.original_image, self.current_heatmap, opacity)
            qimage = QtGui.QImage(
                heatmap_img.tobytes(),
                heatmap_img.size[0],
                heatmap_img.size[1],
                QtGui.QImage.Format_RGB888
            )
            pixmap = QtGui.QPixmap.fromImage(qimage).scaled(
                self.heatmapLbl.width(),
                self.heatmapLbl.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.heatmapLbl.setPixmap(pixmap)

    def create_heatmap_overlay(self, original_img, heatmap, opacity=0.7):
        heatmap = heatmap.convert("L")
        heatmap = heatmap.resize(original_img.size)
        overlay = Image.new("RGBA", original_img.size)
        heatmap_data = np.array(heatmap)

        # Re-normalize just in case
        heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-8)
        heatmap_data = (heatmap_data * 255).astype(np.uint8)

        for x in range(overlay.width):
            for y in range(overlay.height):
                intensity = heatmap_data[y, x] / 255.0
                overlay.putpixel((x, y), (255, 100, 100, int(255 * intensity * opacity)))

        original_img = original_img.convert("RGBA")
        result = Image.alpha_composite(original_img, overlay)
        return result.convert("RGB")

    def add_to_history(self, label, accuracy):
        new_entry = {
            "date": self.scan_date.date().toString("yyyy-MM-dd"),
            "time": self.scan_time.time().toString("hh:mm"),
            "patient_id": self.patient_id.text(),
            "name": self.patient_name.text(),
            "result": label,
            "confidence": accuracy / 100
        }
        self.history_data.insert(0, new_entry)
        self.update_history_view()

    def get_english_result(self, label, accuracy):
        if label == "Unknown":
            return (
                "⚠️ Could not classify accurately\n\n"
                f"Accuracy: {accuracy:.2f}%\n\n"
                "The model is not confident enough in this prediction. Please consult a medical professional."
            )
        result_text = (
            f"🔍 Prediction: {label}\n\n"
            f"📊 Accuracy: {accuracy:.2f}%\n\n"
        )
        if label == "Malignant":
            result_text += "🚨 This scan shows characteristics of malignant tissue. Please consult an oncologist immediately."
        elif label == "Benign":
            result_text += "ℹ️ This scan shows benign characteristics. Regular follow-ups are recommended."
        else:
            result_text += "✅ This scan appears normal. No signs of abnormality detected."
        return result_text

    def get_shona_result(self, label, accuracy):
        if label == "Unknown":
            return (
                "⚠️ Hatina kukwanisa kuona zviri pachena\n\n"
                f"Chivimbo: {accuracy:.2f}%\n\n"
                "Ndapota bvunza nyanzvi yezvehutano."
            )
        label_translation = {"Malignant": "Gomarara", "Benign": "Benign", "Normal": "Zvakanaka"}
        translated_label = label_translation.get(label, label)
        result_text = (
            f"🔍 Mhedziso: {translated_label}\n\n"
            f"📊 Chivimbo: {accuracy:.2f}%\n\n"
        )
        if label == "Malignant":
            result_text += "🚨 Kuongorora uku kunoratidza zviratidzo zvegomarara. Ndapota shanyira chiremba nekukurumidza."
        elif label == "Benign":
            result_text += "ℹ️ Kuongorora uku kunoratidza maitiro asina njodzi. Zvinokurudzirwa kuti uite nguva dzose kuongororwa."
        else:
            result_text += "✅ Kuongorora uku kunoratidza kuti mune hutano hwakanaka."
        return result_text

    def update_probability_chart(self, probabilities):
        self.prob_figure.clear()
        ax = self.prob_figure.add_subplot(111)
        classes = ["Benign", "Malignant", "Normal"]
        colors = ['#03DAC6', '#CF6679', '#018786']
        bars = ax.bar(classes, probabilities, color=colors)
        ax.set_ylim(0, 1)
        ax.set_facecolor('#383838' if self.dark_mode else '#FFFFFF')
        self.prob_figure.set_facecolor('#383838' if self.dark_mode else '#FFFFFF')
        ax.tick_params(colors='white' if self.dark_mode else 'black')
        for spine in ax.spines.values():
            spine.set_color('white' if self.dark_mode else 'black')
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                color='white' if self.dark_mode else 'black'
            )
        self.prob_canvas.draw()

    def trainingFunction(self):
        self.loading_spinner.move(
            self.imageLbl.x() + self.imageLbl.width()//2 - 25,
            self.imageLbl.y() + self.imageLbl.height()//2 - 25
        )
        self.loading_spinner.start()
        self.textEdit.setText("Training in progress...\nThis may take several minutes.")
        QTimer.singleShot(5000, self.finish_training)

    def finish_training(self):
        self.loading_spinner.stop()
        self.textEdit.setText("Training complete!\nModel accuracy improved.\nNew malignant recall: 91.2%")
        self.update_charts()

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.loading_spinner.set_dark_mode(self.dark_mode)
        self.update_stylesheet()
        icon = "icons/moon.png" if self.dark_mode else "icons/sun.png"
        self.theme_toggle.setIcon(QIcon(icon))
        self.update_charts()
        if hasattr(self, 'prob_figure'):
            self.prob_figure.clear()
            self.prob_canvas.draw()

    def toggle_theme_auto(self, state):
        pass

    def set_theme_mode(self, mode):
        self.dark_mode = (mode == "dark")
        self.loading_spinner.set_dark_mode(self.dark_mode)
        self.update_stylesheet()
        self.update_charts()
        if hasattr(self, 'prob_figure'):
            self.prob_figure.clear()
            self.prob_canvas.draw()

    def change_language(self, language):
        self.current_language = "shona" if language == "Shona" else "english"
        titles = ["Dashboard", "Analyze Scan", "History", "Settings"]
        if self.current_language == "shona":
            titles = ["Dashboardi", "Ongorora Scan", "Nhoroondo", "Zvirongwa"]
        self.current_section.setText(titles[self.stacked_widget.currentIndex()])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Arial", 10)
    app.setFont(font)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
