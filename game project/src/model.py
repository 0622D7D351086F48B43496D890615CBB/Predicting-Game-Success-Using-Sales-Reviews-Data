from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def train_model(df):
    """Train Random Forest model to predict meta_score"""

    # ----------------------------------
    # 1. Check target column
    # ----------------------------------
    if 'meta_score' not in df.columns:
        print("❌ 'meta_score' not found")
        return None

    # ----------------------------------
    # 2. Select numeric features
    # ----------------------------------
    features = df.select_dtypes(include=["int64", "float64"]).drop(columns=['meta_score'])
    target = df['meta_score']

    # ----------------------------------
    # 3. Scaling (IMPORTANT)
    # ----------------------------------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    # ----------------------------------
    # 4. Train-Test Split
    # ----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, target, test_size=0.2, random_state=42
    )

    # ----------------------------------
    # 5. Train Random Forest Model
    # ----------------------------------
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    # ----------------------------------
    # 6. Predictions
    # ----------------------------------
    predictions = model.predict(X_test)

    # ----------------------------------
    # 7. Regression Metrics
    # ----------------------------------
    print("\n📊 Model Performance (Regression)")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("R² Score:", r2_score(y_test, predictions))

    # ----------------------------------
    # 8. Convert to Classes for Accuracy
    # ----------------------------------
    def score_to_class(x):
        if x < 50:
            return "Low"
        elif x < 75:
            return "Medium"
        else:
            return "High"

    y_test_class = y_test.apply(score_to_class)
    pred_class = np.array([score_to_class(p) for p in predictions])

    acc = accuracy_score(y_test_class, pred_class)

    print("\n📊 Classification-style Accuracy")
    print("Accuracy Score:", acc)

    return model
