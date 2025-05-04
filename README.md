# SHL Grammar Scoring Engine

This repository contains my solution for the SHL Hiring Assessment hosted on Kaggle. The goal is to develop a Grammar Scoring Engine that predicts a continuous grammar score (between 0 and 5) based on audio recordings.

## Competition Overview

- Develop a model that evaluates grammar usage in spoken audio clips.
- Predict a MOS Likert Grammar Score for each audio file.
- Audio files are between 45 and 60 seconds long.
- Training data: 444 samples. Testing data: 195 samples.

## Evaluation Metric

- Pearson Correlation between predicted and actual scores.

## Dataset

- train.csv: file names and grammar scores
- test.csv: file names (no labels)
- sample_submission.csv: sample format for submission
- Audio files: .wav format

All files should be placed in the `data/` directory.

## Folder Structure

- `data/`: All audio files and CSVs
- `notebooks/`: Jupyter notebooks for EDA, training, and final submission
- `src/`: Python modules for preprocessing, feature extraction, and modeling
- `submissions/`: Final submission files

## Setup

Install required packages:
```bash
pip install -r requirements.txt
