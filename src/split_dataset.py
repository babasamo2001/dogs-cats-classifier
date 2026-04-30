import os
import shutil
import random

random.seed(42)

RAW_DIR = "data/raw"

BASE_DIR = "data/cats_and_dogs"

TRAIN_SPLIT = 0.8


def create_dirs():

    paths = [

        "train/cats",
        "train/dogs",
        "validation/cats",
        "validation/dogs"

    ]

    for path in paths:

        os.makedirs(
            os.path.join(BASE_DIR, path),
            exist_ok=True
        )


def split_class(class_name):

    src_dir = os.path.join(
        RAW_DIR,
        class_name
    )

    images = os.listdir(src_dir)

    random.shuffle(images)

    split_index = int(
        len(images) * TRAIN_SPLIT
    )

    train_imgs = images[:split_index]
    val_imgs = images[split_index:]

    copy_images(
        train_imgs,
        class_name,
        "train"
    )

    copy_images(
        val_imgs,
        class_name,
        "validation"
    )


def copy_images(images, class_name, split):

    dest_dir = os.path.join(
        BASE_DIR,
        split,
        class_name
    )

    src_dir = os.path.join(
        RAW_DIR,
        class_name
    )

    for img in images:

        src = os.path.join(
            src_dir,
            img
        )

        dst = os.path.join(
            dest_dir,
            img
        )

        shutil.copyfile(src, dst)


if __name__ == "__main__":

    create_dirs()

    split_class("cats")
    split_class("dogs")

    print("Dataset split completed.")