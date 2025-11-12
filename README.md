# Jupiter Notebooks

AI, ML, DL, NLP and GAN exploration playground.

## Kernel

- Run `uv sync` to create a venv.
- In Visual Studio Code, from "Select Kernel" in the top-right corner, choose "Python environments..", then select the venv name created in the previous step.

## Environment

### Ubuntu

- Add the following to `~/.config/pip/pip.conf`:

```
[global]
break-system-packages = true
```

## Applications

1. Heart Disease Prediction
   - https://archive.ics.uci.edu/dataset/45/heart+disease
   - https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset
2. Bulldozer sale price prediction
   - https://www.kaggle.com/competitions/bluebook-for-bulldozers/overview
3. Dog Breed Multi-class Classification
   - https://www.kaggle.com/competitions/dog-breed-identification/overview

## Model Diagnostics

### Tensorboard

- `uv run tensorboard --logdir <path>`
