#!/usr/bin/env python3
"""
ibscnn.py
  IBS-CNN, a deep learning model that converts VCF files into images for kinship classification
  using computer vision and Identity-by-State theory.

Usage example:
  python ibscnn.py --image(-i) [image_path] --model(-m) [model.pth] --output(-o) [output_path]
"""

import importlib
import sys
import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import csv


def check_modules():

    required = {
        "torch": "torch",
        "torchvision": "torchvision",
        "PIL": "Pillow",
    }

    missing = []
    for module, pip_name in required.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(" Following dependencies are missing: ")
        print("  pip install " + " ".join(missing))
        sys.exit(1)

# Basic config
label_map = {
    'FS': 0,
    'UN': 1,
    'PO': 2,
    '2nd': 3,
    '3rd': 4,
    '4th': 5,
    '5th': 6
}


inv_label_map = {v: k for k, v in label_map.items()}


fig_width = 128
fig_height = 512


class LongBarcodeCNN(torch.nn.Module):


    def __init__(self, num_classes=7):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),

            torch.nn.Conv2d(32, 64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),

            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_model(model_path, device):

    model = LongBarcodeCNN(num_classes=7)

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    return model


def get_preprocessing_transform():

    return transforms.Compose([
        transforms.Resize((fig_width, fig_height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_images(model, image_dir, transform, device):

    results = []

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith('.png'):
            continue

        filepath = os.path.join(image_dir, filename)

        try:
            image = Image.open(filepath).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                probs = probabilities.cpu().numpy()[0]
                pred_label_idx = torch.argmax(probabilities).item()
                pred_label = inv_label_map[pred_label_idx]

            results.append({
                'filename': filename,
                'pred_label': pred_label,
                'probabilities': probs
            })

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

            results.append({
                'filename': filename,
                'pred_label': 'ERROR',
                'probabilities': [0.0] * len(label_map)
            })

    return results


def save_results(results, output_path):

    header = ['filename', 'pred_label']
    header.extend([f'prob_{inv_label_map[i]}' for i in range(len(label_map))])

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for result in results:
            row = [result['filename'], result['pred_label']]
            row.extend([f"{p:.6f}" for p in result['probabilities']])
            writer.writerow(row)

    print(f"Results saved to: {output_path}")


def main():

    check_modules()

    parser = argparse.ArgumentParser(description='Predict barcode classes from PNG images')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='Path to directory containing PNG images')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Path to trained model (.pth file)')
    parser.add_argument('-o', '--output', type=str, default='predictions.csv',
                        help='Output csv file path (default: predictions.csv), do not forget the suffix!')

    args = parser.parse_args()

    if not os.path.isdir(args.image):
        raise ValueError(f"Image directory not found: {args.image}")

    if not os.path.isfile(args.model):
        raise ValueError(f"Model file not found: {args.model}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model from: {args.model}")
    model = load_model(args.model, device)

    transform = get_preprocessing_transform()

    print(f"Predicting images in: {args.image}")
    results = predict_images(model, args.image, transform, device)

    save_results(results, args.output)

    print(f"Prediction completed. Processed {len(results)} images.")


if __name__ == '__main__':
    main()