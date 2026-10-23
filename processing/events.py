from uuid import uuid5, NAMESPACE_URL
import math


def finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def build_event(frame, prediction, run_id, metadata=None):
    if prediction["label"] not in ("pothole","speed_breaker"):return None
    metadata=metadata or {"source":"synthetic","dataset_id":"legacy-demo","name":"Synthetic demo"}
    impact=frame.loc[frame.az.abs().idxmax()]
    key=f"{run_id}:{metadata.get('dataset_id','unknown')}:{int(frame.window_id.iloc[0])}"
    source=metadata["source"]
    return {"id":str(uuid5(NAMESPACE_URL,key)),"run_id":run_id,"window_id":int(frame.window_id.iloc[0]),
        "dataset_id":metadata.get("dataset_id"),"dataset_name":metadata.get("name"),
        "timestamp":float(impact.timestamp),"latitude":finite(impact.get("latitude")),"longitude":finite(impact.get("longitude")),
        "speed_kmh":finite(impact.get("speed_kmh")),"prediction":prediction["label"],"confidence":prediction["confidence"],
        "method":prediction["method"],"source":source,"model_source":"synthetic",
        "evidence":[],"evidence_note":"No camera evidence supplied.","features":{k:finite(v) for k,v in prediction.get("features",{}).items()}}
