"""
Step 1: Split 70 images into 56 train / 14 test
        and partition annotations.csv + labels.csv accordingly.
"""

import os
import random
import shutil
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
UPLOAD_DIR   = r"data/images"         # folder with all 70 images
ANN_CSV      = r"data/annotations.csv"
LAB_CSV      = r"data/labels.csv"
OUT_ROOT     = r"data"                  # outputs written under data/

TRAIN_IMG    = os.path.join(OUT_ROOT, "train", "images")
TEST_IMG     = os.path.join(OUT_ROOT, "test",  "images")

for d in [TRAIN_IMG, TEST_IMG]:
    os.makedirs(d, exist_ok=True)

RANDOM_SEED = 42

# ── Load CSVs ──────────────────────────────────────────────────────────────────
ann = pd.read_csv(ANN_CSV)
lab = pd.read_csv(LAB_CSV)

# Normalise image column: strip extension and lowercase for matching
ann_images = ann["image"].unique().tolist()          # e.g. "2659ffa5-color.png"
lab_images = lab["image"].unique().tolist()          # e.g. "2659ffa5-color.jpg"

# Use annotation image list as the 70-image ground truth
all_images = sorted(ann_images)
assert len(all_images) == 70, f"Expected 70 images, got {len(all_images)}"

random.seed(RANDOM_SEED)
random.shuffle(all_images)

train_images = all_images[:56]
test_images  = all_images[56:]

print(f"Train: {len(train_images)}  |  Test: {len(test_images)}")

# ── Save split lists ───────────────────────────────────────────────────────────
with open(os.path.join(OUT_ROOT, "train_images.txt"), "w") as f:
    f.write("\n".join(train_images))
with open(os.path.join(OUT_ROOT, "test_images.txt"), "w") as f:
    f.write("\n".join(test_images))

# ── Split annotation CSV ───────────────────────────────────────────────────────
train_ann = ann[ann["image"].isin(train_images)].reset_index(drop=True)
test_ann  = ann[ann["image"].isin(test_images)].reset_index(drop=True)

train_ann.to_csv(os.path.join(OUT_ROOT, "train_annotations.csv"), index=False)
test_ann.to_csv(os.path.join(OUT_ROOT, "test_annotations.csv"),  index=False)
print(f"annotations  → train: {len(train_ann)} rows | test: {len(test_ann)} rows")

# ── Split labels CSV ───────────────────────────────────────────────────────────
# labels.csv uses .jpg; build a stem→lab_image map
stem_to_lab = {os.path.splitext(p)[0]: p for p in lab_images}
stem_to_ann = {os.path.splitext(p)[0]: p for p in ann_images}

train_stems = {os.path.splitext(p)[0] for p in train_images}
test_stems  = {os.path.splitext(p)[0] for p in test_images}

train_lab = lab[lab["image"].apply(lambda p: os.path.splitext(p)[0]).isin(train_stems)].reset_index(drop=True)
test_lab  = lab[lab["image"].apply(lambda p: os.path.splitext(p)[0]).isin(test_stems)].reset_index(drop=True)

train_lab.to_csv(os.path.join(OUT_ROOT, "train_labels.csv"), index=False)
test_lab.to_csv(os.path.join(OUT_ROOT, "test_labels.csv"),  index=False)
print(f"labels       → train: {len(train_lab)} rows | test: {len(test_lab)} rows")

# ── Copy images to train / test folders ───────────────────────────────────────
def copy_images(image_list, dest_folder, src_folder=UPLOAD_DIR):
    copied, missing = 0, []
    for img_name in image_list:
        # Try both .png and .jpg variants
        for ext in [os.path.splitext(img_name)[1], ".png", ".jpg"]:
            stem  = os.path.splitext(img_name)[0]
            src   = os.path.join(src_folder, stem + ext)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest_folder, stem + ext))
                copied += 1
                break
        else:
            missing.append(img_name)
    return copied, missing

c, m = copy_images(train_images, TRAIN_IMG)
print(f"Copied {c} train images  | missing: {m if m else 'none'}")
c, m = copy_images(test_images, TEST_IMG)
print(f"Copied {c} test images   | missing: {m if m else 'none'}")

print("\n✅ Step 1 complete.")
