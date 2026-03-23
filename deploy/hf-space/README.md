---
title: Skin Lesion Defused IBR
emoji: false
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.22.0
app_file: deploy/hf-space/app.py
pinned: false
---

# Skin Lesion Classifier (Defused IBR5 + IBR6)

This Hugging Face Space serves single-image inference for the TensorFlow Defused IBR model.

## Required Space Variables

- MODEL_REPO_ID: Hugging Face model repo id, for example `username/skin-lesion-defused-ibr`
- MODEL_FILENAME: model artifact filename in that repo (default: `best_model.keras`)
- HF_TOKEN: required only if the model repo is private

## Input Contract

- Image format: RGB
- Resize: 224x224
- Normalization: pixel / 255.0
- Classes: nv, mel, bkl, bcc, akiec, df, vasc

## Local Run

```bash
pip install -r deploy/hf-space/requirements.txt
set MODEL_REPO_ID=your-username/your-model-repo
python deploy/hf-space/app.py
```
