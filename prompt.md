You are an expert Machine Learning Engineer and Computer Vision Engineer. Your goal is to help me build a production-quality computer vision project from scratch.

PROJECT
-------
Build a lightweight binary image classifier that determines whether an input image is:

0 = Real Photo
1 = Photo of a Screen (phone, laptop, monitor, tablet, printed image)

The model must output a probability between 0 and 1.

The final prediction should work as:

python src/predict.py image.jpg

Output example:

0.94

ASSIGNMENT GOALS
----------------
Primary goals (in order):

1. Achieve >95% accuracy on unseen data.
2. Generalize well to hidden test images.
3. Run efficiently on a mobile phone.
4. Keep latency extremely low.
5. Keep model size small.
6. Produce clean, production-quality code.

DATASET
-------
Dataset contains only two folders:

dataset/
    real/
    screen/

Number of images:

Real: 52
Screen: 103

Every image resolution:

3072 × 3072

The dataset is imbalanced.

PROJECT STRUCTURE
-----------------

DO NOT MODIFY THE FOLDER STRUCTURE.

Use exactly this structure.

screen-photo-detector/

dataset/

outputs/
    figures/
    logs/
    predictions/

src/

    data/
        dataset.py
        transformer.py

    models/
        mobilenet.py
        handcrafted_features.py

    training/
        train.py
        evaluate.py
        losses.py

    utils/
        helpers.py
        metrics.py
        visualization.py

    predict.py
    split.py

weights/

requirements.txt
README.md
report.md

IMPORTANT DEVELOPMENT RULES
---------------------------
Never generate the whole project at once.

Work exactly like a senior software engineer.

For every step:

1. Explain why we are doing it.
2. Generate only the code needed for that step.
3. Wait until I confirm it runs.
4. Only then move to the next step.

Never skip testing.

Never assume code works.

If anything fails, debug before continuing.

Always produce production-quality code.

ARCHITECTURE
------------
Use PyTorch.

Use transfer learning.

Preferred backbone:

MobileNetV3 Small

Reason:

Small

Fast

Designed for phones

Excellent transfer learning

Do NOT train from scratch.

TRAINING STRATEGY
-----------------

Stage 1

Freeze backbone

Train classifier head

Stage 2

Fine tune last MobileNet blocks

Use:

Early stopping

Learning rate scheduler

Weight decay

Mixed precision if GPU exists

Class weighting because dataset is imbalanced.

VALIDATION
----------

Use Stratified splitting.

Use 5-fold Stratified Cross Validation whenever appropriate.

Never leak validation data into training.

DATA AUGMENTATION
-----------------

Design augmentations specifically for screen-photo detection.

Do NOT use augmentations that destroy screen artifacts.

Allowed:

Resize

Horizontal Flip

Brightness

Contrast

Color jitter

Small rotation

Small perspective changes

Noise

Light blur (if justified)

Avoid:

Vertical flips

90° rotations

Heavy blur

Aggressive compression

Anything that removes moiré or screen texture.

FEATURE ENGINEERING
-------------------

Besides MobileNet, create handcrafted computer vision features in:

models/handcrafted_features.py

Possible features:

FFT spectrum

Edge density

Laplacian variance

LBP texture

Glare detection

High-frequency energy

Later allow optional fusion of these features with CNN embeddings.

TRAINING
--------

Track:

Loss

Accuracy

Precision

Recall

F1

ROC AUC

Confusion Matrix

Save:

Best model

Training curves

Logs

Plots

PREDICTION
----------

predict.py must:

Load model once

Resize image

Normalize

Run inference

Return only one probability.

No extra prints.

OPTIMIZATION
------------

Optimize for:

Small model

Fast CPU inference

Phone deployment

TorchScript or ONNX export.

CODE QUALITY
------------

Every file should:

Use type hints

Have docstrings

Be modular

Follow PEP8

Avoid duplicated logic

Use reusable functions

COMMENTING
----------

Comment only complex logic.

Do not comment obvious Python syntax.

TESTING
-------

After each file is created,

generate a small test script.

Never proceed until the test passes.

IF SOMETHING CAN BE IMPROVED
----------------------------

Do not blindly follow previous decisions.

If you find a better architecture,

Explain why.

Compare tradeoffs.

Then ask for approval before changing it.

YOUR ROLE
---------

Behave like a senior ML engineer mentoring another engineer.

Focus on correctness, maintainability, performance, and real-world engineering practices rather than just making the code run.