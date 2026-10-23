from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
DATABASE = ROOT / ".runtime" / "reports.sqlite3"
CLASSES = ["normal", "pothole", "speed_breaker", "rough_road"]
SAMPLE_RATE = 50
WINDOW_SECONDS = 2
FEATURES = ["peak", "peak_to_peak", "rms", "variance", "gyro_rms", "active_fraction", "sign_changes"]
STATUSES = ["Inspection required", "Confirmed", "Rejected", "Repair scheduled", "Repaired"]
