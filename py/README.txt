Система распознавания автомобильных номеров
Автоматическое обнаружение и распознавание автомобильных номеров на изображениях с использованием YOLOv8 и EasyOCR.

📌 Обзор
Этот проект представляет собой систему на Python для обнаружения и распознавания автомобильных номеров.

Ключевые возможности:

Обнаружение номеров с помощью YOLOv8

Распознавание текста через EasyOCR (поддержка русского и английского)

Улучшение качества изображения перед распознаванием

Проверка и форматирование номеров по стандарту

Визуализация результатов (рамки вокруг номеров + текст)

Экспорт результатов в JSON

✨ Особенности
✅ Детекция номеров – YOLOv8 находит номера на фото/изображениях
✅ Распознавание текста – EasyOCR извлекает текст с поддержкой кириллицы
✅ Улучшение изображения – применяется:

Переход в оттенки серого

Билатеральный фильтр

CLAHE

Адаптивная пороговая обработка

Морфологические операции (удаление шумов)
✅ Проверка номера – автоматическая проверка на соответствие формату
✅ Визуализация – сохранение обработанных изображений с выделенными номерами
✅ Экспорт данных – все результаты сохраняются в JSON

🚀 Установка
Необходимые компоненты
Python 3.8+

CUDA (для работы на GPU, необязательно, но рекомендуется)

Инструкция
Клонируйте репозиторий

Установите зависимости

Поместите изображения в папку images

Поддерживаемые форматы: JPG, JPEG, PNG

Запустите систему


📂 Структура проекта
project/  
├── images/          		# Исходные изображения для обработки  
├── output/          		# Обработанные изображения с номерами  
├── results.json     		# Результаты распознавания  
├── car_plate_detector.py   # Основной скрипт  
└── yolov8n.pt 				# YOLOv8  

📊 Пример работы
Входное изображение:
Пример автомобиля

Результат:

Обнаруженный номер: А123БВ777

Координаты: [x: 120, y: 240, width: 200, height: 80]

Уверенность: 0.92

Визуализация:
Результат с bounding box



*README написанно с использованием нейросетей.