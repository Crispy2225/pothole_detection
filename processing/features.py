import numpy as np
from processing.preprocess import preprocess


def extract(frame):
    clean = preprocess(frame)
    z = clean.az.to_numpy()
    g = clean[["gx", "gy", "gz"]].to_numpy()
    return {"peak": float(np.max(np.abs(z))), "peak_to_peak": float(np.ptp(z)),
            "rms": float(np.sqrt(np.mean(z*z))), "variance": float(np.var(z)),
            "gyro_rms": float(np.sqrt(np.mean(g*g))),
            "active_fraction": float(np.mean(np.abs(z)>1)),
            "sign_changes": float(np.mean(np.diff(np.sign(z)) != 0)),
            "speed_kmh": float(clean.speed_kmh.mean()) if "speed_kmh" in clean and clean.speed_kmh.notna().all() else float("nan")}
