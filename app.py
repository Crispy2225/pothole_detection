"""RoadWatch: local sensor recording analysis and road-event review."""
import json
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from config import ROOT, DATA, MODELS, RESULTS, STATUSES
from simulation.import_csv import import_csv, Recording
from processing.detector import Detector
from processing.events import build_event
from storage.database import save_event, list_events, update_status
from ui.maps import map_html, geojson, valid_location
from ui.insights import replay_summary
from ui.workspace import signal_chart, route_points, value

st.set_page_config(page_title="RoadWatch | Road condition review",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{padding-top:2.5rem;max-width:1500px;padding-bottom:3rem}
h1{font-size:2rem!important;font-weight:650!important;letter-spacing:-.04em}
h2{font-size:1.25rem!important;letter-spacing:-.02em}h3{font-size:1.05rem!important}
[data-testid=stMetricLabel]{font-size:.8rem;opacity:.75}
[data-testid=stMetricValue]{font-size:1.7rem;letter-spacing:-.035em}
[data-testid=stMetric]{border-bottom:1px solid rgba(125,125,125,.23);padding:12px 0 18px}
[data-testid=stSidebar] .block-container{padding:1.5rem}
[data-testid=stSidebar] [role=radiogroup]{gap:.2rem}
[data-testid=stSidebar] label{padding:.25rem 0}
</style>""",unsafe_allow_html=True)

@st.cache_resource
def detectors(version):
    return {name:Detector(name) for name in ("threshold","motion","speed")}

@st.cache_data
def demo(version):
    raw=(DATA/"scenarios/trip_20.csv").read_bytes()
    df=pd.read_csv(DATA/"scenarios/trip_20.csv")
    return Recording(df,[part.reset_index(drop=True) for _,part in df.groupby("window_id")],
        {"name":"Vehicle response simulation", "source":"synthetic","dataset_id":sha256(raw).hexdigest(),
         "samples":len(df),"sample_rate_hz":50,"duration_s":len(df)/50,"has_speed":True,
         "gps_coverage":float(df.latitude.notna().mean()),"labelled_windows":df.window_id.nunique(),
         "warnings":[],"generator_version":2})


def reset_run():
    st.session_state.update(run_id=str(uuid4()),cursor=0,playing=False,history=[],last=None)

if not (MODELS/"metadata.json").exists():
    st.error("Models are missing. Run python bootstrap.py from the project folder.")
    st.stop()
model_version=(MODELS/"speed.pkl").stat().st_mtime_ns
models=detectors(model_version)
if "run_id" not in st.session_state:reset_run()
with st.sidebar:
    st.markdown("## RoadWatch")
    st.caption("ROAD CONDITION REVIEW")
    page=st.radio("Workspace",["Trip workspace","Map & reports","Investigation","Model validation","Import recording","System notes"],label_visibility="collapsed",key="page",disabled=st.session_state.playing)
    st.divider()
    imports_dir=ROOT/".runtime/imports"
    saved_imports={}
    if imports_dir.exists():
        for meta_path in sorted(imports_dir.glob("*.json")):
            try:
                meta=json.loads(meta_path.read_text());saved_imports[meta["dataset_id"]]=meta
            except (ValueError,KeyError):continue
    options=["demo"]+list(saved_imports)
    if "pending_dataset" in st.session_state:st.session_state.dataset_choice=st.session_state.pop("pending_dataset")
    if st.session_state.get("dataset_choice","demo") not in options:st.session_state.dataset_choice="demo"
    chosen=st.selectbox("Active recording",options,format_func=lambda k:"Built-in simulation" if k=="demo" else saved_imports[k]["name"],key="dataset_choice",disabled=st.session_state.playing)
    st.caption("Local storage · no sensor stream required")

if chosen=="demo":
    recording=demo((DATA/"scenarios/trip_20.csv").stat().st_mtime_ns)
else:
    meta=saved_imports[chosen]
    raw=(imports_dir/f"{chosen}.csv").read_bytes()
    units=meta.get("units",{})
    recording=import_csv(raw,meta["name"],units.get("timestamp","seconds"),units.get("acceleration","m/s²"),units.get("gyro","rad/s"),meta["source"])
windows=recording.windows;metadata=recording.metadata
if st.session_state.get("active_id")!=metadata["dataset_id"]:
    reset_run();st.session_state.active_id=metadata["dataset_id"]
source_label="SIMULATION" if metadata["source"]=="synthetic" else "IMPORTED RECORDING"
with st.sidebar:
    st.divider()
    allowed=["motion","threshold"] if not metadata["has_speed"] else ["speed","motion","threshold"]
    if st.session_state.get("method") not in allowed:st.session_state.method=allowed[0]
    method=st.selectbox("Detector",allowed,format_func=lambda k:{"speed":"Random Forest · motion + speed","motion":"Random Forest · motion","threshold":"Acceleration threshold"}[k],key="method",disabled=st.session_state.playing)
    if not metadata["has_speed"]:st.caption("Speed unavailable: motion detectors remain available.")
    interval=st.select_slider("Playback interval",[.5,1.0,2.0],value=1.0,disabled=st.session_state.playing)
    st.caption("Models trained on synthetic signals. Imported-data predictions require field validation.")

st.caption(f"WORKSPACE / {page.upper()}")
left,right=st.columns([4,1])
left.title(page)
right.caption(source_label)
right.write(f"{len(windows)} windows · {metadata['sample_rate_hz']:.0f} Hz")
st.caption(metadata["name"]+" · "+("Artificial vehicle-like signals; not a measured road trip." if metadata["source"]=="synthetic" else "User-supplied sensor recording. Labels and GPS are not independently verified."))


def advance():
    index=st.session_state.cursor
    if index>=len(windows):st.session_state.playing=False;return
    frame=windows[index];prediction=models[method].predict(frame)
    event=build_event(frame,prediction,st.session_state.run_id,metadata)
    if event:save_event(event)
    truth=frame.label.iloc[0] if "label" in frame else None
    if pd.isna(truth):truth=None
    st.session_state.history.append({"Window":index,"Time (s)":float(frame.timestamp.iloc[0]),"Ground truth":truth,"Prediction":prediction["label"],"Probability":prediction["confidence"],"Method":method})
    st.session_state.last=(index,prediction);st.session_state.cursor+=1
    if st.session_state.cursor==len(windows):st.session_state.playing=False


if page=="Trip workspace":
    controls=st.columns([1,1,1.3,1.4,1.3])
    if controls[0].button("Play",type="primary",disabled=st.session_state.cursor>=len(windows) or st.session_state.playing):
        st.session_state.playing=True;st.rerun()
    if controls[1].button("Pause",disabled=not st.session_state.playing):
        st.session_state.playing=False;st.rerun()
    if controls[2].button("Next window",disabled=st.session_state.playing or st.session_state.cursor>=len(windows)):advance()
    if controls[3].button("Analyze full trip",disabled=st.session_state.playing or st.session_state.cursor>=len(windows)):
        with st.spinner("Processing remaining windows…"):
            while st.session_state.cursor<len(windows):advance()
    if controls[4].button("New analysis",disabled=st.session_state.playing):reset_run();st.rerun()

    @st.fragment(run_every=interval if st.session_state.playing else None)
    def playback():
        was_playing=st.session_state.playing
        if was_playing:advance()
        reports=[e for e in list_events() if e["run_id"]==st.session_state.run_id]
        cols=st.columns(4)
        cols[0].metric("Recording duration",f"{metadata['duration_s']:.0f} s")
        cols[1].metric("Analysis progress",f"{st.session_state.cursor} / {len(windows)}")
        cols[2].metric("Suspected events",len(reports))
        cols[3].metric("GPS coverage",f"{metadata['gps_coverage']:.0%}")
        st.progress(st.session_state.cursor/len(windows))
        main,detail=st.columns([3,1])
        with main:
            st.subheader("Vertical motion")
            if st.session_state.last:
                idx,prediction=st.session_state.last
                frame=windows[idx]
                signal_chart(frame)
            else:
                prediction=None
                signal_chart(recording.frame)
                st.caption("Full recording preview. Play or step through two-second windows to run detection.")
        with detail:
            st.subheader("Latest decision")
            if prediction:
                st.write(prediction["label"].replace("_"," ").title())
                st.caption("Threshold decision" if prediction["confidence"] is None else f"Model probability {prediction['confidence']:.1%} · uncalibrated")
                st.metric("Peak acceleration",value(prediction["features"]["peak"]," m/s²"))
                st.metric("RMS",value(prediction["features"]["rms"]," m/s²"))
            else:st.write("Awaiting analysis")
            st.caption("A suspected event is a review candidate, not a confirmed road defect.")
        st.subheader("Window ledger")
        if st.session_state.history:st.dataframe(pd.DataFrame(st.session_state.history),hide_index=True,width="stretch")
        else:st.caption("Predictions appear here as the recording is processed.")
        if was_playing and not st.session_state.playing:st.rerun()
    playback()

elif page=="Map & reports":
    all_reports=list_events()
    f1,f2=st.columns([1,2])
    scope=f1.selectbox("Recording scope",["Active recording","All stored reports"])
    statuses=f2.multiselect("Review states",STATUSES,default=STATUSES)
    reports=[e for e in all_reports if e["status"] in statuses and (scope=="All stored reports" or e.get("dataset_id")==metadata["dataset_id"])]
    categories=st.multiselect("Detected categories",["pothole","speed_breaker"],default=["pothole","speed_breaker"],format_func=lambda k:k.replace("_"," ").title())
    reports=[e for e in reports if e.get("prediction") in categories]
    valid=[e for e in reports if valid_location(e)]
    cols=st.columns(3)
    cols[0].metric("Reports",len(reports));cols[1].metric("Located observations",len(valid));cols[2].metric("Awaiting inspection",sum(e["status"]=="Inspection required" for e in reports))
    tiles=st.toggle("Street basemap",value=True,help="Street tiles require internet. Turn off for offline coordinate markers.")
    selected=st.selectbox("Inspect report",[None]+[e["id"] for e in reports],format_func=lambda k:"Overview" if k is None else k[:8])
    trajectory=route_points(recording.frame) if scope=="Active recording" else []
    if valid or any(point is not None for point in trajectory):
        components.html(map_html(valid,trajectory,selected,tiles,metadata["source"] if scope=="Active recording" else "mixed"),height=505)
    else:
        st.info("No coordinates available for this view. Sensor analysis and unlocated report review remain available.")
    st.caption("Popup photos are category-matched samples from your image folders, not photographs of the detected location.")
    st.caption("Pin number = observations at the same coordinate. Color = selected or newest observation's review state. Synthetic paths may cross buildings and do not represent actual roads.")
    if len(valid)<len(reports):st.caption(f"{len(reports)-len(valid)} report(s) have no GPS fix and remain in the ledger without map pins.")
    if reports:
        st.dataframe(pd.DataFrame(reports)[["id","source","dataset_name","timestamp","prediction","latitude","longitude","method","status"]] if all("dataset_name" in e for e in reports) else pd.DataFrame(reports)[["id","source","timestamp","prediction","latitude","longitude","method","status"]],hide_index=True,width="stretch")
        export1,export2=st.columns(2)
        export1.download_button("Export reports · JSON",json.dumps(reports,indent=2,allow_nan=False),"reports.json","application/json")
        export2.download_button("Export locations · GeoJSON",json.dumps(geojson(valid),indent=2,allow_nan=False),"locations.geojson","application/geo+json")
        if selected:
            event=next(e for e in reports if e["id"]==selected)
            st.subheader("Report "+selected[:8])
            detail,review=st.columns([2,1])
            detail.write(f"Trip time: {event['timestamp']:.2f} s · Speed: {value(event.get('speed_kmh'),' km/h')}")
            detail.caption(event.get("evidence_note","Legacy simulation illustration references; not photographs."))
            status=review.selectbox("Review status",STATUSES,index=STATUSES.index(event["status"]),key="status_"+selected)
            if review.button("Save review"):
                update_status(selected,status);st.rerun()
            with st.expander("Stored event data"):st.json(event)
    else:st.info("No matching reports. Analyze a recording, or change the recording scope and review filters.")

elif page=="Investigation":
    history=st.session_state.history;summary=replay_summary(history)
    a,b,c,d=st.columns(4)
    a.metric("Labelled / processed",f"{summary['labelled']} / {len(history)}")
    b.metric("False alarms",summary["false_positives"] if summary["labelled"] else "—")
    c.metric("Missed potholes",summary["false_negatives"] if summary["labelled"] else "—")
    d.metric("Pothole precision",f"{summary['precision']:.1%}" if summary["precision"] is not None else "—")
    st.caption("Outcomes use only windows with complete labels. Unlabelled recordings do not receive accuracy metrics. A run may combine detectors if you switch methods.")
    idx=st.slider("Inspect window",0,len(windows)-1,min(max(st.session_state.cursor-1,0),len(windows)-1))
    frame=windows[idx]
    cols=st.columns([3,1])
    with cols[0]:signal_chart(frame,channels=("ax","ay","az"))
    with cols[1]:
        label=frame.label.iloc[0] if "label" in frame else None
        st.write("Reference label")
        st.write("Not supplied / ambiguous" if pd.isna(label) else label.replace("_"," ").title())
        st.metric("Mean speed",value(frame.speed_kmh.mean()," km/h"))
        st.caption("Window inspection runs inference without saving reports.")
    rows=[]
    for name,detector in models.items():
        if name=="speed" and not frame.speed_kmh.notna().all():
            rows.append({"Detector":name,"Prediction":"Unavailable: missing speed","Probability":None});continue
        prediction=detector.predict(frame)
        rows.append({"Detector":name,"Prediction":prediction["label"],"Probability":prediction["confidence"]})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    usable=[row for row in rows if not row["Prediction"].startswith("Unavailable")]
    if len({row["Prediction"]=="pothole" for row in usable})>1:st.warning("Detectors disagree on whether this window contains a pothole.")
    feature=models["motion"].predict(frame)["features"]
    st.caption(f"Peak {feature['peak']:.2f} m/s² · Threshold {models['threshold'].threshold:.2f} m/s² · RMS {feature['rms']:.2f} m/s²")
    with st.expander("Global feature importance"):
        bundle=models["speed"].bundle
        st.bar_chart(pd.Series(bundle["model"].feature_importances_,index=bundle["features"],name="Importance"))
        st.caption("Training-set impurity importance. Not a causal explanation or evidence of real-road reliability.")
    if history:
        st.subheader("Analysis ledger")
        st.dataframe(pd.DataFrame(history),hide_index=True,width="stretch")
        st.download_button("Export window predictions",pd.DataFrame(history).to_csv(index=False),"window_predictions.csv","text/csv")

elif page=="Model validation":
    result=json.loads((RESULTS/"metrics.json").read_text())
    st.subheader("Held-out simulation benchmark")
    st.caption("These results describe the bundled synthetic dataset, regardless of the active recording. Imported files are not used to train or tune these models.")
    a,b,c=st.columns(3)
    a.metric("Test windows",result["test_windows"]);b.metric("Held-out trips",len(result["test_trips"]));c.metric("Generator", "Version 2")
    table=pd.DataFrame({k:{m:v for m,v in values.items() if m!="confusion_matrix"} for k,values in result["models"].items()}).T
    st.dataframe(table.style.format("{:.3f}"),width="stretch")
    st.bar_chart(table[["precision","recall","f1"]])
    for name in ("motion","speed"):
        st.subheader(name.title()+" · confusion matrix")
        st.caption("Rows: actual class. Columns: predicted class.")
        st.dataframe(pd.DataFrame(result["models"][name]["confusion_matrix"],index=result["class_order"],columns=result["class_order"]),width="stretch")
    st.info("More varied simulation is a software stress test, not a substitute for independently labelled vehicle recordings.")

elif page=="Import recording":
    st.subheader("Bring a vehicle recording into the workspace")
    if st.session_state.pop("import_saved",False):st.success("Recording saved and selected. Open Trip workspace to analyze it.")
    st.write("One CSV per trip, sampled at 50 Hz. Required columns are timestamp, ax, ay, az, gx, gy, gz. Speed, GPS and labels are optional.")
    upload=st.file_uploader("Sensor CSV",type=["csv"],help="Maximum 10 MB / 100,000 samples. Imported files are stored locally when you select Save recording.")
    c1,c2,c3=st.columns(3)
    time_unit=c1.selectbox("Timestamp units",["seconds","milliseconds"])
    acc_unit=c2.selectbox("Acceleration units",["m/s²","g"])
    gyro_unit=c3.selectbox("Angular velocity units",["rad/s","deg/s"])
    provenance=st.radio("Recording origin",["Vehicle / recorded sensors","Synthetic / generated"],horizontal=True)
    orientation=st.checkbox("Acceleration is gravity-removed, z is vertical, and the sensor mounting is fixed.")
    st.caption("Raw accelerometer values including gravity require calibration/orientation compensation before import. This declaration cannot be verified automatically.")
    if upload:
        try:
            imported=import_csv(upload.getvalue(),Path(upload.name).name,time_unit,acc_unit,gyro_unit,"imported" if provenance.startswith("Vehicle") else "synthetic")
            a,b,c=st.columns(3)
            a.metric("Valid windows",len(imported.windows));b.metric("GPS coverage",f"{imported.metadata['gps_coverage']:.0%}");c.metric("Labelled windows",imported.metadata["labelled_windows"])
            for warning in imported.metadata["warnings"]:st.warning(warning)
            st.dataframe(imported.frame.head(15),hide_index=True,width="stretch")
            if st.button("Save recording",type="primary",disabled=not orientation):
                imports_dir.mkdir(parents=True,exist_ok=True)
                key=imported.metadata["dataset_id"]
                (imports_dir/f"{key}.csv").write_bytes(upload.getvalue())
                (imports_dir/f"{key}.json").write_text(json.dumps(imported.metadata,indent=2))
                st.session_state.pending_dataset=key
                st.session_state.import_saved=True
                st.rerun()
        except ValueError as exc:st.error(str(exc))
    st.divider()
    st.subheader("Logger format")
    st.code("timestamp,ax,ay,az,gx,gy,gz,speed_kmh,latitude,longitude,label",language=None)
    st.write("Use elapsed seconds or milliseconds; acceleration in m/s² or g; angular velocity in rad/s or deg/s. Use speed_kmh for speed, latitude/longitude in decimal degrees. Leave optional values blank when unavailable.")
    st.caption("Optional labels: normal, pothole, speed_breaker, rough_road. Mixed or partially labelled windows are excluded from accuracy calculations. Multi-trip files, invalid values, timing gaps and unsupported sample rates are rejected with an explanation.")
    sample=(DATA/"scenarios/trip_20.csv").read_bytes()
    st.download_button("Download synthetic format example",sample,"synthetic_format_example.csv","text/csv")
    st.caption("The example is generated data; choose Synthetic / generated when importing it. It is not a road recording.")

else:
    st.subheader("Processing contract")
    st.code("""CSV / simulation → validation + unit conversion → 100-sample windows
→ smoothing → shared features → detector → suspected-event record
→ SQLite → coordinate map + review workflow""")
    st.write("The current model expects gravity-removed vertical acceleration at 50 Hz. Complete speed is required only for the speed-aware model. GPS absence never blocks motion analysis; it produces an unlocated report. No camera images are fabricated for imported events.")
    st.subheader("Simulation version 2")
    st.write("Continuous speed and braking/turning components; imbalanced randomized road classes; variable pulse width, rebound and suspension response; noisy 1 Hz GPS fixes with occasional loss. The path integrates speed and heading but is not snapped to a surveyed road. This is an illustrative signal generator, not a validated vehicle dynamics model.")
    st.subheader("Storage and model provenance")
    st.write("Imported CSVs and their unit/provenance metadata are saved locally under .runtime/imports. Models remain trained on separate synthetic trips. Reports retain dataset identifiers, source and model provenance. Import does not retrain a model. Previously saved reports remain available under All stored reports.")
    st.caption("See README.md for the file overview, architecture, CSV contract, limitations, and test commands.")
