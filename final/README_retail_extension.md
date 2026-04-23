# Real-World Retail Clothing Extension

This folder extends `assignment2.ipynb` with a real-world clothing image pipeline. The workflow keeps the ten Fashion-MNIST labels but trains a transfer-learning model on retailer images.

## 0. Archive an Old Scrape

Before rebuilding the dataset, archive the current active folders:

```bash
python scripts/archive_retail_dataset.py
```

This moves the active `raw/`, `cropped/`, `segmented/`, `segmentation_overlays/`, and `metadata/` folders under `data/retail_images/archive/<date>_tops_pilot/`, then recreates clean active folders.

## 1. Add Retailer Sources

Edit `configs/retailer_sources.csv` as needed. Keep `fashion_mnist_class` as one of:

`T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`

Use multiple rows per class. The downloader skips blank rows, blocked pages, unreadable images, very small images, and non-image responses. Some category URLs will fail or expose no images; that is expected with retailer sites.

## 2. Download Images

Dry-run candidate extraction first:

```bash
python scripts/scrape_retail_images.py --dry-run
```

Then download up to 150 usable images per class:

```bash
python scripts/scrape_retail_images.py
```

Outputs:

- `data/retail_images/raw/<class_slug>/...`
- `data/retail_images/metadata/downloads.csv`

## 3. Segment, Validate, and Split

Retailer images often include a full person, background, pose, and other visual context. Before validation, create garment-focused segmented copies with the Clothes SegFormer parser:

```bash
python scripts/segment_retail_images.py
```

Segmented training images are written to `data/retail_images/segmented/<class_slug>/...`, and QA overlays are written to `data/retail_images/segmentation_overlays/<class_slug>/...`. Review the overlays before training.

For a small segmentation smoke test:

```bash
python scripts/segment_retail_images.py --limit-per-class 10 --overwrite
```

The older heuristic cropper is still available as a fallback:

```bash
python scripts/crop_retail_images.py --mode foreground --overwrite
```

If either crop mode looks worse than the original photos, use full-image copies instead:

```bash
python scripts/crop_retail_images.py --mode copy --overwrite
```

```bash
python scripts/validate_retail_dataset.py
```

Outputs:

- `data/retail_images/metadata/validated_images.csv`
- `data/retail_images/metadata/splits.csv`

## 4. Generate EDA Figures

```bash
python scripts/make_retail_eda.py
```

Outputs are written to `results/retail_finetune/`.

## 5. Fine-Tune MobileNetV3

Run a smoke test after validation:

```bash
python scripts/train_retail_finetune.py --smoke --skip-fashionmnist-eval
```

Run the full transfer-learning experiment:

```bash
python scripts/train_retail_finetune.py
```

Outputs:

- `results/retail_finetune/training_history.csv`
- `results/retail_finetune/metrics_summary.json`
- `results/retail_finetune/classification_report.csv`
- `results/retail_finetune/confusion_matrix.png`
- `results/retail_finetune/training_curve.png`
- `results/retail_finetune/mobilenetv3_retail_finetuned.pt`

## 6. Build the Paper PDF

```bash
python scripts/build_report.py
```

Output:

- `report/retail_finetuning_paper.pdf`

The PDF can be rendered before data collection, but metric fields and figures will be marked pending until the scrape, validation, EDA, and training scripts have been run.
