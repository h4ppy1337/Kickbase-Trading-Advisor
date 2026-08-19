from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_model(X_train, y_train):
    """Train a RandomForestRegressor model, parameters optimized via grid search"""

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model using RMSE, MAE, R2 and percentage of correct sign predictions"""

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    signs_correct = np.sum(np.sign(y_test) == np.sign(y_pred))
    signs_percent = (signs_correct / len(y_test)) * 100

    return signs_percent, rmse, mae, r2
