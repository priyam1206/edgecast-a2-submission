from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Forecast:
    predicted_arrivals: Dict[str, float]
    confidence: Dict[str, float]
    trend: Dict[str, float]


class ShortHorizonPredictor:
    """
    Lightweight trend-based predictor.

    It intentionally uses only past arrivals. The demo keeps the predictor
    simple so the methodology is easy to explain and compare.
    """

    def __init__(self, window: int = 5, horizon: int = 3):
        if window < 3:
            raise ValueError("window must be at least 3")
        self.window = window
        self.horizon = horizon

    def forecast(self, history: Dict[str, List[int]]) -> Forecast:
        predicted, confidence, trend = {}, {}, {}

        for model, values in history.items():
            recent = values[-self.window:]
            if not recent:
                predicted[model] = 0.0
                confidence[model] = 0.0
                trend[model] = 0.0
                continue

            if len(recent) == 1:
                predicted[model] = float(recent[-1])
                confidence[model] = 0.0
                trend[model] = 0.0
                continue

            # Least-squares slope over the recent window.
            n = len(recent)
            xs = list(range(n))
            x_mean = sum(xs) / n
            y_mean = sum(recent) / n
            denom = sum((x - x_mean) ** 2 for x in xs) or 1.0
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, recent)) / denom

            pred = max(0.0, recent[-1] + slope * self.horizon)

            # Confidence = trend consistency. 1.0 means the recent step
            # directions mostly agree with the estimated trend.
            diffs = [b - a for a, b in zip(recent[:-1], recent[1:])]
            if abs(slope) < 1e-9:
                consistency = 1.0 if all(abs(d) < 1e-9 for d in diffs) else 0.5
            else:
                expected_sign = 1 if slope > 0 else -1
                agree = sum(1 for d in diffs if (d > 0 and expected_sign > 0) or (d < 0 and expected_sign < 0) or d == 0)
                consistency = agree / len(diffs)

            predicted[model] = pred
            confidence[model] = consistency
            trend[model] = slope

        return Forecast(predicted, confidence, trend)
