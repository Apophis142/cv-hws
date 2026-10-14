# Hough Transform Pipeline

Детекция окружностей на изображениях с помощью градиентного преобразования Хафа.

## Что делает

- Принимает на вход файл или директорию с изображениями (sRGB).
- Для каждого изображения:
  - выделяет края через Кэнни,
  - находит окружности через Hough с нормировкой голосов на `2πr`,
  - сохраняет края, бинарную маску и overlay в `output_dir`.
- Пишет `report.txt` или `report.md` с числом найденных окружностей по каждому файлу.

## Запуск

```bash
python -m tasks.hough_transform.main <input_path> -o <output_dir> [опции]
```

`input_path` — файл или директория. Для директории обход рекурсивный,
поддерживаются `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`.

Пример:

```bash
python -m tasks.hough_transform.main ./data/logs_pipes -o ./out/hough \
    --r-min 0.03 --r-max 0.15 \
    --r-delta 0.005 --ab-delta 0.005 \
    --threshold 0.3 --min-dist 0.02
```

## Параметры

Параметры
Canny:

- `--canny-low`, `--canny-high` — пороги гистерезиса в `[0, 1]`, по умолчанию `0.08` / `0.17`.

Hough:

- `--r-min`, `--r-max` — диапазон радиусов.

- `--r-delta` — шаг по радиусу.

- `--ab-delta` — шаг бинаризации центра (влияет на память и скорость).

- `--threshold` — минимальная доля видимой дуги `2πr`, в `(0, 1]`.

- `--min-dist` — минимальное расстояние между центрами.

- `--max-circles` — верхняя граница числа окружностей на изображение.

- `--report-format` — `txt` или `md`.

Все геометрические параметры (`r-min`, `r-max`, `r-delta`, `ab-delta`, `min-dist`)
можно задавать либо в пикселях (int), либо как долю от `min(H, W)` (`float` из `(0, 1)`).
Например, --`r-min` `0.03` при `min(H, W) = 600` даст `18` пикселей.

## Результат

```
<output_dir>/
├── report.txt
├── edges/
│   └──<name>_canny.png
├── masks/
│   └── <name>_mask.png
└── overlays/
    └── <name>_overlay.png
```
- `edges` — бинарное изображение, белые границы на чёрном фоне, найденные оператором Кэнни.  
- `mask` — бинарное изображение, белые окружности на чёрном фоне.
- `overlay` — исходное изображение с выделенными и подписанными контурами.

## Параметры для изображений в `data/logs_pipes/`

```bash
python -m tasks.hough_transform.main .\data\logs_pipes\logs-2.jpg -o tasks/hough_transform/output --canny-low 0.15 --canny-high 0.35 --r-delta 1 --threshold 0.09 --min-dist 0.05 --max-circles 100 --r-min 0.035
python -m tasks.hough_transform.main .\data\logs_pipes\logs-2.jpg -o tasks/hough_transform/output --canny-low 0.15 --canny-high 0.35 --r-delta 1 --threshold 0.09 --min-dist 0.05 --max-circles 100 --r-min 0.035
python -m tasks.hough_transform.main .\data\logs_pipes\logs-3.jpg -o tasks/hough_transform/output --canny-low 0.25 --canny-high 0.4 --r-delta 1 --threshold 0.15 --min-dist 0.025 --max-circles 200 --r-min 0.01 --r-max 0.02
python -m tasks.hough_transform.main .\data\logs_pipes\logs-4.jpg -o tasks/hough_transform/output --canny-low 0.25 --canny-high 0.5 --r-delta 1 --threshold 0.125 --min-dist 0.05 --max-circles 200 --r-min 0.025 --r-max 0.06
python -m tasks.hough_transform.main .\data\logs_pipes\pipes-4.jpg -o tasks/hough_transform/output --canny-low 0.5 --canny-high 0.6 --r-delta 1 --threshold 0.125 --min-dist 0.06 --max-circles 100 --r-min 0.03 --r-max 0.1
python -m tasks.hough_transform.main .\data\logs_pipes\pipes-5.jpg -o tasks/hough_transform/output --canny-low 0.2 --canny-high 0.4 --r-delta 1 --threshold 0.1 --min-dist 0.025 --max-circles 200 --r-min 0.02 --r-max 0.22
python -m tasks.hough_transform.main .\data\logs_pipes\pipes-6.jpg -o tasks/hough_transform/output --canny-low 0.3 --canny-high 0.5 --r-delta 1 --threshold 0.1 --min-dist 0.03 --max-circles 200 --r-min 0.03 --r-max 0.25з
```