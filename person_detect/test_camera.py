import cv2
from ultralytics import YOLO

def detect_people_from_camera():
    # Загрузка модели YOLOv8
    model = YOLO('./yolov8n_ncnn_model', task = 'detect')  # Замените на нужную модель (например, 'yolov8s.pt' для версии small)

    # Открытие USB камеры
    cap = cv2.VideoCapture('/dev/video0')  # Используйте 0 для первой подключенной камеры
    if not cap.isOpened():
        print("Ошибка открытия камеры")
        return

    # Установите разрешение камеры (например, 640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры")
            break

        # Детектирование объектов на текущем кадре
        results = model(frame)

        # Обработка результатов детектирования
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Получение координат бокса и класса объекта
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = box.cls
                if cls == 0:  # класс 0 обычно обозначает "человека"
                    # Рисование бокса на кадре
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Показ кадра
        cv2.imshow('USB Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

# Запуск функции для обработки видео с камеры
detect_people_from_camera()
