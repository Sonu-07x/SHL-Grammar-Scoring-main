from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def train_model(X_train, y_train):
    model = Ridge()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    score = pearsonr(preds, y_val)[0]
    return score
