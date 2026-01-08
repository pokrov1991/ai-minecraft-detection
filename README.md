# Детекция персонажей в Minecraft: FCOS vs YOLO

## Описание проекта

Цель данного проекта — сравнить две современные модели детекции объектов  
**FCOS (Fully Convolutional One-Stage Detector)** и **YOLOv8**  
на задаче обнаружения персонажей и мобов в игре **Minecraft**.

Проект выполняется в рамках учебного модуля и охватывает **полный ML-pipeline**:
- анализ и подготовку датасета,
- обучение моделей,
- инференс на изображениях и видео,
- оценку и сравнение метрик качества и скорости,
- формирование итогового отчёта.

---

## Цели проекта

1. Подготовить датасет с аннотациями в формате **VOC XML** к форматам,
   пригодным для обучения:
   - **COCO JSON** (для MMDetection / FCOS),
   - **YOLO txt + data.yaml** (для Ultralytics YOLO).
2. Обучить модель **FCOS** с использованием фреймворка **MMDetection**.
3. Обучить модель **YOLOv8** с использованием библиотеки **Ultralytics**.
4. Выполнить инференс:
   - на отдельных изображениях,
   - на видеофайле из мира Minecraft.
5. Сравнить модели по следующим метрикам:
   - `mAP`
   - `mAP@0.5`
   - `Precision`
   - `Recall`
   - `F1-score`
   - `FPS` (скорость инференса на видео)
6. Сформировать итоговые артефакты и PDF-отчёт.

---

## Структура проекта

```
mmdetection/
│
├── datasets/
│   ├── minecraft/                     # исходный датасет (VOC XML)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── minecraft_prepared/             # подготовленные данные
│       ├── coco/                       # COCO формат (для FCOS)
│       │   ├── train.json
│       │   ├── val.json
│       │   ├── test.json
│       │   └── images/
│       │       ├── train/
│       │       ├── val/
│       │       └── test/
│       │
│       └── yolo/                       # YOLO формат
│           ├── images/
│           │   ├── train/
│           │   ├── val/
│           │   └── test/
│           ├── labels/
│           │   ├── train/
│           │   ├── val/
│           │   └── test/
│           └── data.yaml
│
├── configs/
│   └── fcos/
│       └── fcos_minecraft.py           # конфиг FCOS под Minecraft
│
├── scripts/
│   └── prepare_minecraft_dataset.py    # XML → COCO + YOLO
│
├── artifacts/
│   ├── fcos/                           # чекпоинты и логи FCOS
│   ├── yolo/                           # чекпоинты и логи YOLO
│   ├── inference/
│   │   ├── fcos/                       # инференс изображений (FCOS)
│   │   └── yolo/                       # инференс изображений (YOLO)
│   ├── videos/
│   │   ├── fcos_inference.mp4
│   │   └── yolo_inference.mp4
│   ├── metrics/
│   │   └── metrics_comparison.csv      # итоговое сравнение метрик
│   └── report.pdf                      # финальный отчёт
│
├── notebook.ipynb                      # основной ноутбук проекта
└── README.md
