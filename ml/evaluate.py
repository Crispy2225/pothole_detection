import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from config import DATA, RESULTS, CLASSES
from processing.detector import Detector
import pandas as pd


def main():
    manifest = json.loads((DATA/"manifest.json").read_text())
    windows=[]
    for i in manifest["splits"]["test"]:
        df=pd.read_csv(DATA/"scenarios"/f"trip_{i:02d}.csv")
        windows.extend(part for _,part in df.groupby("window_id"))
    truth=[part.label.iloc[0] for part in windows]
    result={"source":"synthetic", "evaluation_unit":"non-overlapping 2-second window", "test_trips":manifest["splits"]["test"], "test_windows":len(windows), "class_order":CLASSES, "models":{}}
    for method in ["threshold","motion","speed"]:
        detector=Detector(method)
        predicted=[detector.predict(part)["label"] for part in windows]
        p,r,f,_=precision_recall_fscore_support([v=="pothole" for v in truth],[v=="pothole" for v in predicted],average="binary",zero_division=0)
        result["models"][method]={"precision":float(p),"recall":float(r),"f1":float(f)}
        if method != "threshold":
            result["models"][method]["confusion_matrix"]=confusion_matrix(truth,predicted,labels=CLASSES).tolist()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/"metrics.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
