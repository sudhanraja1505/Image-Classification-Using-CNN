"""Train and evaluate the CNN on CIFAR-10.
Usage:  python train.py                 (30 epochs)
        python train.py --epochs 5      (quick test)"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")                    # save plots to files; no window needed
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from model import build_model, CLASSES


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return x_train, y_train.ravel(), x_test, y_test.ravel()


def plot_curves(history, path="training_curves.png"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key, title in zip(ax, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        a.plot(history.history[key], label="train")
        a.plot(history.history["val_" + key], label="validation")
        a.set_title(title); a.set_xlabel("Epoch"); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_confusion(cm, path="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test set)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="cifar10_cnn.keras")
    args = p.parse_args()

    keras.utils.set_random_seed(42)
    x_train, y_train, x_test, y_test = load_data()
    idx = np.random.RandomState(42).permutation(len(x_train))      # hold out 10% for validation
    x_train, y_train = x_train[idx], y_train[idx]
    x_val, y_val = x_train[:5000], y_train[:5000]
    x_train, y_train = x_train[5000:], y_train[5000:]
    print(f"train {len(x_train)} | validation {len(x_val)} | test {len(x_test)}")

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5),
    ]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}   Test loss: {loss:.4f}")
    y_pred = model.predict(x_test, verbose=0).argmax(axis=1)
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=3))

    plot_curves(history)
    plot_confusion(confusion_matrix(y_test, y_pred))
    model.save(args.out)
    print(f"Saved model to {args.out}, plots to training_curves.png and confusion_matrix.png")


if __name__ == "__main__":
    main()
