import yaml, json, os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.audio_utils import extract_features
from src.model import train_model, evaluate_model
import joblib

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    config = load_config("experiments/exp1_mfcc_ridge/config.yaml")

    df = pd.read_csv("dataset/train.csv")
    X = [extract_features(f"dataset/audios_train/{f}", sr=config["sample_rate"], n_mfcc=config["n_mfcc"]) for f in df["file_name"]]
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])
    model = train_model(X_train, y_train)
    score = evaluate_model(model, X_val, y_val)

    out_dir = f"experiments/{config['experiment_name']}"
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, f"{out_dir}/model.pkl")
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump({"pearson_correlation": score}, f)

if __name__ == "__main__":
    main()