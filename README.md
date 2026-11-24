# Cats & Dogs Object Detection

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![torchvision](https://img.shields.io/badge/torchvision-Faster%20R--CNN-EE4C2C)

Detecting and classifying cats and dogs in images. I built a simple two-head detector on top of ResNet-50 from scratch, then compared it with a fine-tuned Faster R-CNN.

Coursework, Financial University (Moscow).

## Results

Custom two-head ResNet-50 model, after 20 epochs on a held-out test set (110 images):

| Metric | Test |
|---|---|
| Classification accuracy (cat vs dog) | **94.5%** |
| Box accuracy (IoU ≥ 0.5) | **89.1%** |

## Approach

1. **Dataset:** 1,100 images of cats and dogs with Pascal VOC XML annotations, split 90/10 into train and test sets. I wrote a custom `AnimalDetectionDataset` that parses the XML files and returns the image, the bounding box and the label.
2. **Custom model:** a pretrained ResNet-50 backbone with two heads. One head regresses the bounding box coordinates, the other classifies the animal. The loss is the sum of MSE (box) and cross-entropy (class).
3. **Faster R-CNN:** fine-tuned `fasterrcnn_resnet50_fpn` from torchvision by replacing the box predictor with a `FastRCNNPredictor` for the two classes.
4. **Visualization:** plotted ground-truth and predicted boxes on test images.

## Tech stack

Python · PyTorch · torchvision · Pillow · Matplotlib · tqdm

## How to run

```bash
pip install torch torchvision matplotlib pillow xmltodict tqdm
```

Place `animals.zip` (images and XML annotations) next to the notebook and run `Детекция.ipynb`. A GPU is recommended.
