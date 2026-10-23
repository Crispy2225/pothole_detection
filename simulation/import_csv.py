"""Strict, bounded CSV import. No invented GPS, labels, speed or evidence."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO, StringIO
import csv
import numpy as np
import pandas as pd
from config import CLASSES, SAMPLE_RATE, WINDOW_SECONDS
from processing.preprocess import validate, MOTION

@dataclass
class Recording:
    frame: pd.DataFrame
    windows: list
    metadata: dict


def import_csv(raw, name="recording.csv", time_unit="seconds", acceleration_unit="m/s²", gyro_unit="rad/s", source="imported"):
    if not raw or len(raw)>10*1024*1024:
        raise ValueError("Choose a nonempty CSV smaller than 10 MB.")
    if source not in ("imported","synthetic"):
        raise ValueError("Choose recorded or synthetic provenance.")
    try:
        header=next(csv.reader(StringIO(raw.decode("utf-8-sig"))))
        normalized=[name.strip().lower() for name in header]
        if len(set(normalized))!=len(normalized):raise ValueError("Duplicate column names in CSV header.")
    except (UnicodeError,StopIteration,csv.Error) as exc:
        raise ValueError("Use a UTF-8 CSV with a valid header.") from exc
    try:
        df=pd.read_csv(BytesIO(raw),nrows=100001)
    except (ValueError,UnicodeError) as exc:
        raise ValueError("Unable to read CSV. Use UTF-8 text with a comma-separated header.") from exc
    if len(df)>100000:
        raise ValueError("Maximum 100,000 rows per recording. Split larger trips before import.")
    df.columns=df.columns.str.strip().str.lower()
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names after normalization.")
    missing=set(MOTION)-set(df.columns)
    if missing: raise ValueError("Missing columns: "+", ".join(sorted(missing)))
    if "trip_id" in df and df.trip_id.nunique(dropna=False)>1:
        raise ValueError("Import one trip per file; this CSV contains multiple trip_id values.")
    for col in MOTION+["speed_kmh","latitude","longitude"]:
        if col not in df: df[col]=np.nan
        try: df[col]=pd.to_numeric(df[col],errors="raise")
        except (ValueError,TypeError) as exc: raise ValueError(f"{col} must be numeric. Timestamps must be elapsed seconds or milliseconds.") from exc
    if time_unit not in ("seconds","milliseconds") or acceleration_unit not in ("m/s²","g") or gyro_unit not in ("rad/s","deg/s"):
        raise ValueError("Unsupported unit selection.")
    if time_unit=="milliseconds": df["timestamp"]/=1000
    if acceleration_unit=="g": df[["ax","ay","az"]]*=9.80665
    if gyro_unit=="deg/s": df[["gx","gy","gz"]]*=np.pi/180
    validate(df)
    origin=float(df.timestamp.iloc[0]);df["timestamp"]-=origin
    intervals=np.diff(df.timestamp)
    rate=1/np.median(intervals)
    # Keep preprocessing identical to training; don't silently resample hardware signals.
    if not 47.5<=rate<=52.5:
        raise ValueError(f"Observed sampling rate is {rate:.1f} Hz. This model requires 50 Hz (±5%). Configure your logger or resample externally before import.")
    if np.max(np.abs(intervals-.02))>.006:
        raise ValueError("Timing gaps or jitter exceed 6 ms from the 20 ms target. Split gaps and resample to 50 Hz before import; missing motion is not interpolated automatically.")
    # A one-sided GPS fix is not a usable location.
    incomplete=df.latitude.isna()|df.longitude.isna()
    df.loc[incomplete,["latitude","longitude"]]=np.nan
    supplied_labels="label" in df
    if not supplied_labels: df["label"]=None
    else:
        df["label"]=df.label.replace(r"^\s*$",None,regex=True)
        bad=set(df.label.dropna())-set(CLASSES)
        if bad: raise ValueError("Unknown labels: "+", ".join(sorted(map(str,bad))))
    count=SAMPLE_RATE*WINDOW_SECONDS
    usable=(len(df)//count)*count
    if usable<count: raise ValueError("At least 100 samples (two seconds at 50 Hz) are required.")
    warnings=[]
    if len(df)>usable: warnings.append(f"Excluded the final {len(df)-usable} samples: they do not form a full 100-sample window.")
    if incomplete.any(): warnings.append(f"{int(incomplete.sum())} samples have no complete GPS fix; events at these samples stay unlocated.")
    if df.speed_kmh.isna().any(): warnings.append("Speed is missing in some samples. Use the motion-only model or threshold detector.")
    df["trip_id"]=0
    df["window_id"]=np.arange(len(df))//count
    windows=[]; ambiguous=0
    for _,part in df.iloc[:usable].groupby("window_id",sort=True):
        part=part.copy().reset_index(drop=True)
        # A window with mixed or partial labels has no single valid ground truth.
        labels=part.label.dropna().unique()
        if len(labels)!=1 or part.label.isna().any():
            part["label"]=None;ambiguous+=1
        windows.append(part)
    if ambiguous: warnings.append(f"{ambiguous} windows have no complete, unambiguous label. They are excluded from accuracy metrics.")
    metadata={"name":name,"source":source,"dataset_id":sha256(raw+repr((time_unit,acceleration_unit,gyro_unit,source)).encode()).hexdigest(),"content_sha256":sha256(raw).hexdigest(),"samples":len(df),"usable_samples":usable,
        "sample_rate_hz":float(rate),"duration_s":float(df.timestamp.iloc[-1]+.02),"timestamp_origin_s":origin,
        "has_speed":bool(df.iloc[:usable].speed_kmh.notna().all()),"gps_coverage":float((~incomplete).mean()),
        "labelled_windows":len(windows)-ambiguous,"warnings":warnings,
        "units":{"timestamp":time_unit,"acceleration":acceleration_unit,"gyro":gyro_unit},
        "assumption":"User declares gravity removed, z vertical, fixed vehicle mounting; not automatically verified."}
    return Recording(df,windows,metadata)
