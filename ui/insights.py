"""Outcome metrics exclude unknown labels; no inferred physical severity."""
def replay_summary(history):
    tp=fp=fn=tn=labelled=0
    for row in history:
        label=row.get("Ground truth",row.get("Ground truth (synthetic)"))
        if label is None or label not in ("normal","pothole","speed_breaker","rough_road"):continue
        labelled+=1
        actual=label=="pothole";predicted=row["Prediction"]=="pothole"
        tp+=actual and predicted;fp+=not actual and predicted;fn+=actual and not predicted;tn+=not actual and not predicted
    return dict(true_positives=tp,false_positives=fp,false_negatives=fn,true_negatives=tn,labelled=labelled,
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)


def report_summary(events):
    from ui.maps import valid_location
    scores=[e["confidence"] for e in events if e.get("confidence") is not None]
    return dict(reports=len(events),runs=len({e["run_id"] for e in events}),
        locations=len({(e["latitude"],e["longitude"]) for e in events if valid_location(e)}),
        pending=sum(e["status"]=="Inspection required" for e in events),probability_count=len(scores),
        mean_probability=sum(scores)/len(scores) if scores else None)
