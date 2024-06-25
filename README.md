# Yolo_RaspberryPi
Проект реализующий портирование нейронной сети на полётный контроллер БПЛА построенный на базе RaspberryPi

## Оглавление
- [Описание проекта](#описание-проекта)
- [Требования](#требования)
- [Установка](#установка)
- [Использование](#использование)
- [Структура проекта](#структура-проекта)
- [Результаты проекта](#результаты-проекта)
- [Вклад](#вклад)
- [Лицензия](#лицензия)
- [Контакты](#контакты)

## Описание проекта
Этот проект посвящен портированию нейронной сети YOLOv8 на полётный контроллер беспилотного летательного аппарата (БПЛА) на базе Raspberry Pi CM4. Основная цель проекта - создать систему распознавания объектов в реальном времени, которая может быть использована для различных приложений, включая навигацию, обнаружение препятствий и мониторинг.

## Требования
Для успешного выполнения этого проекта необходимы следующие компоненты и библиотеки:

### Оборудование
- Raspberry Pi CM4
- Камера, совместимая с Raspberry Pi
- Платформа для БПЛА

### Программное обеспечение
- Raspberry Pi OS
- Python 3.7+
- OpenCV
- PyTorch
- YOLOv8

### Библиотеки Python
Установите необходимые библиотеки:
```sh
pip install torch torchvision torchaudio
pip install opencv-python
pip install yolov8
```

## Установка
Следуйте инструкциям ниже для установки и настройки проекта:
### 1. Клонируйте репозиторий
```sh
git clone https://github.com/username/yolov8-uav-port.git
cd yolov8-uav-port
```
### 2. Установите необходимые библиотеки:
```sh
pip install -r requirements.txt
```
### 3. Настройте окружение:
- Установите Raspberry Pi OS на ваш Raspberry Pi CM4;
- Настройте камеру и убедитесь, что она работает корректно;
- Скопируйте код на Raspberry Pi CM4.

## Использование
Следуйте этим шагам для запуска нейронной сети на полётном контроллере:
### 1. Запустите скрипт распознавания объектов:
```sh
python run_yolov8.py
```
### 2. Просмотрите результаты:
Результаты распознавания объектов будут отображаться в режиме реального времени на экране

### Структура проекта
Описание структуры файлов и папок в проекте:
```bash
yolov8-uav-port/
│
├── data/                   # Данные и модели
│   ├── yolov8.pt           # Предобученная модель YOLOv8
│   └── sample_images/      # Примеры изображений для тестирования
│
├── scripts/                # Скрипты для запуска и настройки
│   ├── run_yolov8.py       # Основной скрипт для запуска YOLOv8
│   └── setup_camera.py     # Скрипт для настройки камеры
│
├── docs/                   # Документация проекта
│   └── README.md           # Основной файл документации
│
├── tests/                  # Тесты для проверки работоспособности
│   └── test_yolov8.py      # Тесты для YOLOv8
│
├── requirements.txt        # Список зависимостей проекта
└── LICENSE                 # Лицензия проекта
```
## Результаты проекта
### Скорость обработки одного кадра используя стандарную модель YOLOv8n

### Скорость обработки одного кадра используя модель YOLOv8n NCNN


## Вклад
Если вы хотите внести вклад в проект, пожалуйста, следуйте этим инструкциям:
1. Форкните репозиторий;
2. Создайте новую ветку (git checkout -b feature/имя_фичи);
3. Внесите свои изменения и закоммитьте их (git commit -m 'Добавил новую фичу');
4. Запушьте изменения в свою ветку (git push origin feature/имя_фичи);
5. Создайте Pull Request.

## Лицензия
Этот проект лицензирован под лицензией MIT. Подробности смотрите в файле LICENSE.

## Контакты
Для вопросов и предложений обращайтесь:
- Автор: Соколов Александр, Кузьмин Захар
- Email: 
- GitHub: 
