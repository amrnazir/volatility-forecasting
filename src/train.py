from sklearn.ensemble import RandomForestRegressor
from src.data import download_data
from src.features import create_features
from src.evaluate import evaluate_predictions


def main():

    # 1. Download historical data
    data = download_data(
        ticker="SPY",
        start="2015-01-01",
        end="2025-01-01"
    )

    # 2. Create features and target
    df = create_features(data)

    features = [
        "volatility_5d",
        "volatility_20d",
        "volatility_60d",
        "volume_change",
    ]

    target = "target_volatility"

    X = df[features]
    y = df[target]

    # 3. Time-based train/test split.
    # Financial data must not be randomly shuffled because
    # that can allow information from the future into training.
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # 4. Naive baseline:
    # predict future volatility using the most recent
    # observed 20-day volatility.
    baseline_predictions = X_test["volatility_20d"]

    baseline_metrics = evaluate_predictions(
        y_test,
        baseline_predictions
    )

    print("Naive baseline:")
    print(f"MAE:  {baseline_metrics['MAE']:.4f}")
    print(f"RMSE: {baseline_metrics['RMSE']:.4f}")

    # 5. Train Random Forest
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 6. Generate predictions
    predictions = model.predict(X_test)

    # 7. Evaluate Random Forest
    model_metrics = evaluate_predictions(
        y_test,
        predictions
    )

    print("\nRandom Forest:")
    print(f"MAE:  {model_metrics['MAE']:.4f}")
    print(f"RMSE: {model_metrics['RMSE']:.4f}")


if __name__ == "__main__":
    main()
