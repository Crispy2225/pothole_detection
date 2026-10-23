"""50 Hz, vehicle-aligned, gravity-removed acceleration input contract."""
import numpy as np
MOTION = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz"]
COLUMNS = MOTION + ["speed_kmh", "latitude", "longitude"]


def validate(frame):
    missing = set(MOTION) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if len(frame)<2 or not np.isfinite(frame[MOTION].to_numpy(dtype=float)).all():
        raise ValueError("Motion channels require at least two finite samples; repair missing sensor readings before import.")
    if (np.diff(frame.timestamp)<=0).any():
        raise ValueError("Timestamps must be strictly increasing. Remove duplicates and order the recording before import.")
    for col,lo,hi in [("speed_kmh",0,300),("latitude",-90,90),("longitude",-180,180)]:
        if col in frame:
            values=frame[col].dropna()
            if not values.between(lo,hi).all():
                raise ValueError(f"{col} contains values outside [{lo}, {hi}]. Check units and coordinates.")
    return frame


def preprocess(frame):
    validate(frame)
    clean=frame.copy()
    for name in MOTION[1:]:
        clean[name]=frame[name].rolling(3,center=True,min_periods=1).mean()
    return clean
