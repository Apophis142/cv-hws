# Traffic Lights Classifier

Классификация сигнала светофора на изображении: `red`, `yellow`, `green`.

## Установка

```bash
pip install -r requirments.txt
```

## Классификация изображений

```bash
python -m tasks.traffic_lights_classifier.main \
    --input data/traffic_lights/images \
    --output tasks/traffic_lights_classifier/answers
```

Опции:

| Опция | По умолчанию | Описание |
|---|---|---|
| `--input` | `data/traffic_lights/images` | Директория с изображениями |
| `--output` | `tasks/traffic_lights_classifier/answers` | Директория для файлов ответов |
| `--config` | — | Путь к JSON с сигнатурами и порогами |
| `--clusters` | `10` | Начальное число кластеров для k-means |
| `-v`, `--verbose` | выкл | Печатать детали по каждому кластеру |

Поддерживаемые форматы: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`.

## Оценка качества

```bash
python -m tasks.traffic_lights_classifier.checker \
    --pred tasks/traffic_lights_classifier/answers \
    --true data/traffic_lights/answers \
    --details
```

Опции:

| Опция | Описание |
|---|---|
| `--pred` | Директория с предсказаниями |
| `--true` | Директория с эталонами |
| `--details` | Показать все объекты с неверным классом |

Вывод: `Accuracy`, `F1-macro`, `F1-micro`, список расхождений.

## Формат ответов

Каждый класс — отдельный `.txt`-файл в директории ответов:

- `red.txt` — красный;
- `yellow.txt` — жёлтый;
- `green.txt` — зелёный.

## Конфигурация

Параметры задаются в `config.py` (`ColorClassifierConfig`) или переопределяются через JSON, переданный в `--config`.

Пример JSON:

```json
{
  "color_signatures": {
    "red":    [{"name": "pure_red", "L": 53, "a": 80, "b": 67}],
    "orange": [{"name": "pure_orange", "L": 65, "a": 40, "b": 65}],
    "yellow": [{"name": "pure_yellow", "L": 95, "a": -10, "b": 90}],
    "green":  [{"name": "pure_green", "L": 85, "a": -80, "b": 80}],
    "blue":   [{"name": "pure_blue", "L": 32, "a": 79, "b": -108}],
    "achromatic": [{"name": "white", "L": 95, "a": 0, "b": 0}]
  },
  "hue_boundaries_deg": {
    "red_orange": 45,
    "orange_yellow": 72,
    "yellow_green": 105
  },
  "blue_hue_range": [180, 320],
  "achromatic_chroma_threshold": 15,
  "disputed_hue_margin": 8,
  "fine_metric": "ciede2000",
  "use_hue_prefilter": true,
  "tau": 0.5,
  "margin": 0.1,
  "min_valid_ratio": 0.0,
  "initial_clusters": 10,
  "max_clusters": 25,
  "cluster_step": 5
}
```

Все поля опциональны: отсутствующие берут значения по умолчанию.