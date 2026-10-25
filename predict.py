"""Classify your own images with the trained model.
Usage:  python predict.py photo1.jpg photo2.png
Note: the model was trained on 32x32 images of 10 classes, so it works best on a
clear, centred photo of one of: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck."""
import argparse
import numpy as np
from PIL import Image
from tensorflow import keras
from model import CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", help="path(s) to image file(s)")
    p.add_argument("--model", default="cifar10_cnn.keras")
    args = p.parse_args()

    model = keras.models.load_model(args.model)
    for path in args.images:
        try:
            img = Image.open(path).convert("RGB").resize((32, 32))
        except Exception as e:
            print(f"{path}: could not open image ({e})"); continue
        probs = model.predict(np.asarray(img, dtype="float32")[None], verbose=0)[0]
        print(f"\n{path}")
        for i in probs.argsort()[::-1][:3]:
            print(f"  {CLASSES[i]:<11} {probs[i] * 100:5.1f}%")


if __name__ == "__main__":
    main()
