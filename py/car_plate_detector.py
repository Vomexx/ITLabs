import cv2
import json
from ultralytics import YOLO
import easyocr
from pathlib import Path
from typing import List, Dict, Any, Optional
import glob
import numpy as np
import re

def initialize_models():
    yolo_model = YOLO('yolov8n.pt')
    ocr_reader = easyocr.Reader(['en', 'ru'])
    return yolo_model, ocr_reader

def preprocess_plate_image(plate_image: np.ndarray) -> np.ndarray:
    """Улучшение изображения номерного знака перед распознаванием"""
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return gray

def is_valid_russian_plate(text: str) -> bool:
    """Проверка соответствия формату российского номерного знака"""
    plate_pattern = re.compile(r'^[АВЕКМНОРСТУХABEKMHOPCTYX]\d{3}[АВЕКМНОРСТУХABEKMHOPCTYX]{2}\d{2,3}$')
    
    clean_text = text.replace(' ', '').upper()
    
    return bool(plate_pattern.match(clean_text))

def postprocess_plate_text(text: str) -> str:
    """Коррекция распознанного текста номерного знака"""
    clean_text = re.sub(r'[^АВЕКМНОРСТУХABEKMHOPCTYX0-9]', '', text.upper())
    
    if len(clean_text) >= 6:
        letters = re.sub(r'[^АВЕКМНОРСТУХABEKMHOPCTYX]', '', clean_text)
        digits = re.sub(r'[^0-9]', '', clean_text)
        
        if len(digits) >= 3 and len(letters) >= 3:
            formatted = f"{letters[0]}{digits[:3]}{letters[1:3]}"
            if len(digits) > 3:
                formatted += digits[3:]
            return formatted
    
    return clean_text

def detect_license_plates(image_path: str, yolo_model) -> List[Dict[str, Any]]:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return []
    
    results = yolo_model(image)
    
    plates = []
    for result in results:
        for box in result.boxes:
            if box.cls == 0:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                
                plates.append({
                    "box": [int(x1), int(y1), int(width), int(height)],
                    "confidence": float(box.conf[0])
                })
    
    return plates

def recognize_plate_text(image_path: str, plate_info: Dict[str, Any], ocr_reader) -> str:
    image = cv2.imread(image_path)
    if image is None:
        return ""
    
    x, y, width, height = plate_info["box"]
    plate_image = image[y:y+height, x:x+width]
    
    processed_plate = preprocess_plate_image(plate_image)
    
    results = ocr_reader.readtext(processed_plate, detail=0)
    raw_text = " ".join(results).strip()
    processed_text = postprocess_plate_text(raw_text)
    if is_valid_russian_plate(processed_text):
        return processed_text
    elif is_valid_russian_plate(raw_text):
        return raw_text
    else:
        return processed_text


def draw_bounding_boxes(image_path: str, plates: List[Dict[str, Any]], output_dir: str = "output") -> Optional[str]:
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    for plate in plates:
        x, y, w, h = plate["box"]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, plate.get("text", ""), (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    Path(output_dir).mkdir(exist_ok=True)
    output_path = f"{output_dir}/{Path(image_path).name}"
    cv2.imwrite(output_path, image)
    return output_path

def process_image(image_path: str, yolo_model, ocr_reader) -> Dict[str, Any]:
    plates = detect_license_plates(image_path, yolo_model)
    
    for plate in plates:
        plate["text"] = recognize_plate_text(image_path, plate, ocr_reader)
    
    output_image_path = draw_bounding_boxes(image_path, plates)
    
    return {
        "filename": Path(image_path).name,
        "plates": plates,
        "output_image": output_image_path
    }

def save_results_to_json(results: List[Dict[str, Any]], output_path: str = "results.json"):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Результаты сохранены в {output_path}")

def get_image_paths(folder: str = "images") -> List[str]:
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob.glob(f"{folder}/{ext}", recursive=True))
        image_paths.extend(glob.glob(f"{folder}/*/{ext}", recursive=True))
    
    return sorted(list(set(image_paths)))

def main():
    if not Path("images").exists():
        print("Ошибка: папка 'images' не найдена")
        return
    
    yolo_model, ocr_reader = initialize_models()
    image_paths = get_image_paths()
    
    if not image_paths:
        print("В папке 'images' не найдено изображений (jpg/jpeg/png)")
        return
    
    print(f"Найдено {len(image_paths)} изображений для обработки...")
    
    results = []
    for image_path in image_paths:
        print(f"\nОбработка {image_path}...")
        result = process_image(image_path, yolo_model, ocr_reader)
        results.append(result)
        
        print(f"Результаты для {Path(image_path).name}:")
        if not result["plates"]:
            print("  Номера не обнаружены")
        else:
            for i, plate in enumerate(result["plates"], 1):
                print(f"  Номер {i}:")
                print(f"    Координаты: {plate['box']}")
                print(f"    Текст: {plate.get('text', 'не распознан')}")
                print(f"    Уверенность: {plate.get('confidence', 0):.2f}")
            print(f"  Результат с bounding boxes сохранен в: {result['output_image']}")
    
    save_results_to_json(results)

if __name__ == "__main__":
    main()