# COOKMATE
Микросервис распознавания продуктов на изображениях (холодильник, стол, тележка/корзинка) на основе ансамбля из двух моделей YOLOv11.

## 📌 Оглавление
- [Структура проекта](#-структура-проекта)
- [Что реализовано](#-что-реализовано)
- [Контракты взаимодействия](#-контракты-взаимодействия)
- [Установка и запуск](#-установка-и-запуск)
- [Тестирование](#-тестирование)
- [Как использовать в качестве строительного блока](#-как-использовать-в-качестве-строительного-блока)

## 📁 Структура проекта

├── `data/`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; # **Архивы с наборами данных (Git LFS) + data.yaml** <br>
├── `instructions/`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; # **Инструкции, использованные при сборе/разметке данных** <br>
├── `models/`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; # **Веса обученных моделей (Git LFS)** <br>
&emsp;&emsp;   ├── `yolo11n_best.pt`&emsp;&emsp;&emsp;   # **YOLOv11 Nano (высокая полнота, conf=0.30)** <br> 
&emsp;&emsp;   ├── `yolo11s_best.pt`&emsp;&emsp;&emsp;   # **YOLOv11 Small (высокая точность, conf=0.75)** <br>
├── `main.py`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; # **FastAPI приложение (ансамбль моделей)** <br>
├── `class_mapping.py`&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp; # **Mapping классов набора данных** <br>
├── `prepare_dataset.py`&emsp;&emsp;&emsp;&ensp;&nbsp; # **Скрипт для подготовки набора данных** <br>
├── `Dockerfile`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp; # **Docker-образ (CPU)** <br>
├── `Dockerfile.gpu`&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp; # **Docker-образ (GPU)** <br>
├── `docker-compose.yml`&emsp;&emsp;&emsp;&ensp;&nbsp; # **Оркестрация контейнеров** <br>
├── `requirements.txt`&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp; # **Зависимости Python** <br>
└── `README.md`&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; # **Этот файл** <br>

<a id="-что-реализовано"></a>
## ⚙️ Что реализовано

1) **Ансамбль моделей YOLOv11**<br><br>
Для балансировки между recall и precision используется стратегия ансамбля:
- YOLOv11 Small (порог уверенности $0.75$) — формирует ядро детекций с минимальным количеством ложных срабатываний;
- YOLOv11 Nano (порог уверенности $0.30$) — находит объекты, которые пропустила Small модель;
- Алгоритм слияния: детекции Small берутся за основу, к ним добавляются детекции Nano, если они не пересекаются с существующими (проверка по IoU > $0.45$).

2) **REST API на FastAPI**<br>
- Асинхронная обработка запросов;
- Endpoint для инференса и проверки здоровья сервиса (Healthcheck).

3) **Контейнеризация**<br>
- Поддержка CPU и GPU (NVIDIA CUDA) через разные Dockerfile;
- Модели не заложены в образ, а подгружаются из внешнего volume для гибкости обновлений.
  
<a id="-контракты-взаимодействия"></a>
## 📡 Контракты взаимодействия

***Предсказание продуктов***

*Запрос:* `POST /predict`<br>

Content-Type: `multipart/form-data`<br>
Body: `file` (изображение в формате JPEG/PNG)<br>

```json
Ответ:
{
  "detections": [
    {
      "class": "milk",
      "confidence": 0.92,
      "bbox": [120.5, 200.3, 350.7, 580.2],
      "source": "small"
    },
    {
      "class": "apple",
      "confidence": 0.45,
      "bbox": [50.2, 400.1, 150.8, 520.5],
      "source": "nano"
    }
  ],
  "count": 2,
  "processing_time": 0.45,
  "stats": {
    "from_small": 1,
    "from_nano_added": 1,
    "total_nano": 2
  }
}
```

***Проверка работоспособности***

*Запрос:* `GET /health`<br>
```json
Ответ:
{
  "status": "ok",
  "models_loaded": true
}
```

<a id="-установка-и-запуск"></a>
## ⚙️ Установка и запуск

*Требования*:
* Docker ≥ 24.0
* Docker Compose v2
* Git LFS

1. **Инициализируйте Git LFS и склонируйте репозиторий**
```bash
git lfs install
git clone https://github.com/v-rjabinin/COOKMATE.git
cd COOKMATE
```
2. **Запустите сервисы**
```bash
# Для CPU (универсальный вариант)
docker compose up -d --build

# Для GPU (требует установленный nvidia-docker)
# Отредактируйте docker-compose.yml, раскомментировав секцию deploy.resources
docker compose up -d --build
```

После запуска API будет доступен по адресу: `http://localhost:8000`

<a id="-тестирование"></a>
## 🧪 Тестирование
*Способ 1*: Swagger UI (в браузере)

1) Откройте `http://localhost:8000/docs`
2) Выберите POST /predict
3) Нажмите "Try it out"
4) Загрузите фото и выполните запрос

*Способ 2*: CURL (в терминале)
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/your/fridge_photo.jpg"
```
*Способ 3*: Python
```python
import requests
with open("path/to/your/fridge_photo.jpg", "rb") as f:
    response = requests.post("http://localhost:8000/predict", files={"file": f})
    print(response.json())
```

<a id="-как-использовать-в-качестве-строительного-блока"></a>
## 📦 Как использовать в качестве строительного блока
1) Добавьте сервис `cookmate-api` в ваш `docker-compose.yml`
2) Убедитесь, что веса моделей (`yolo11n_best.pt`, `yolo11s_best.pt`) доступны в папке `./models` и примонтированы в контейнер
3) Отправляйте HTTP POST-запросы на `http://cookmate-api:8000/predict` с файлом изображения
4) Парсите JSON-ответ для получения списка продуктов и их координат на фото
5) Готов к интеграции в любые системы, использующие HTTP/REST API (мобильные приложения, веб-интерфейсы, бэкенд-сервисы)
