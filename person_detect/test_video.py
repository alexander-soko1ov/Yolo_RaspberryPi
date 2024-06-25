import cv2
from ultralytics import YOLO
import time

def detect_people_from_video():
    # Загрузка модели YOLOv8
    model = YOLO('./yolov8n_ncnn_model', task='detect')  # Замените на нужную модель (например, 'yolov8s.pt' для версии small)

    # Открытие видеофайла
    cap = cv2.VideoCapture('test_2.mp4')  # Замените на путь к вашему видеофайлу
    if not cap.isOpened():
        print("Ошибка открытия видеофайла")
        return

    # Переменные для измерения FPS
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр из видео или видео закончилось")
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

        # Увеличение счетчика FPS
        fps_counter += 1

        # Подсчет FPS
        if time.time() - fps_start_time >= 1:
            fps = round(fps_counter / (time.time() - fps_start_time), 1)
            fps_counter = 0
            fps_start_time = time.time()

        # Вывод FPS на кадр
        cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Показ кадра
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

# Запуск функции для обработки видео
detect_people_from_video()
