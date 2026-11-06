from __future__ import annotations

from typing import Dict, List
import random


def ramp_video_burst(seed: int = 7, slots: int = 70) -> List[Dict[str, int]]:
    """
    A deterministic-by-seed scenario designed for demonstration:
    video demand gradually rises, becomes dominant, then settles.
    """
    rng = random.Random(seed)
    out = []

    for t in range(slots):
        text = max(0, int(round(rng.gauss(5.0, 1.0))))
        image = max(0, int(round(rng.gauss(4.0, 0.9))))

        if t < 18:
            video_mean = 3.5
        elif t < 36:
            video_mean = 3.5 + (t - 18) * 0.75
        elif t < 55:
            video_mean = 17.0
        else:
            video_mean = max(5.0, 17.0 - (t - 55) * 0.7)

        video = max(0, int(round(rng.gauss(video_mean, 1.0))))
        out.append({"text": text, "image": image, "video": video})

    return out


def sudden_spike(seed: int = 11, slots: int = 60) -> List[Dict[str, int]]:
    """
    Abrupt load spike. Useful for showing that predictive logic still needs
    the reactive path because not every burst is forecastable.
    """
    rng = random.Random(seed)
    out = []
    for t in range(slots):
        text = max(0, int(round(rng.gauss(5, 1))))
        image = max(0, int(round(rng.gauss(4, 1))))
        video_mean = 3 if t < 25 else (18 if t < 40 else 5)
        video = max(0, int(round(rng.gauss(video_mean, 1.2))))
        out.append({"text": text, "image": image, "video": video})
    return out
