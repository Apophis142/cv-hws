Изображения в папке должны иметь имена вида `frame_8.jpg`, `frame_9.jpg`, `frame_10.jpg`. Порядок определяется числовым суффиксом; начальный номер произвольный. Соседние кадры должны перекрываться.

Запуск из корня репозитория:

```bash
python -m tasks.panorama_stitching.main \
    --input data/buildings/building1/ \
    --output tasks/panorama_stitching/output/panorama.png
```

Параметры склейки находятся в `config.py`.
