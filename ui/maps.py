"""Leaflet map with local assets and external, attributed street tiles."""
import json
import math
from pathlib import Path
from ui.sample_images import sample_image

COLORS = {"Inspection required":"#cc6b13", "Confirmed":"#c93438", "Rejected":"#737d8c", "Repair scheduled":"#7159ba", "Repaired":"#16806b"}


def valid_location(event):
    try:
        lat,lon=float(event["latitude"]),float(event["longitude"])
        return math.isfinite(lat) and math.isfinite(lon) and -90<=lat<=90 and -180<=lon<=180
    except (KeyError,TypeError,ValueError):
        return False


def geojson(events):
    return {"type":"FeatureCollection", "features":[{"type":"Feature", "geometry":{"type":"Point", "coordinates":[float(e["longitude"]),float(e["latitude"])]}, "properties":{k:v for k,v in e.items() if k not in ("longitude","latitude","evidence")}} for e in events if valid_location(e)]}


def map_html(events, route, selected=None, tiles=True, source="synthetic"):
    vendor=Path(__file__).parent/"vendor"
    display_events=[];images={}
    for event in events:
        if not valid_location(event):continue
        record=dict(event)
        sample=sample_image(event)
        if sample:
            images[sample["key"]]=sample["url"]
            record["sample_image_key"]=sample["key"]
        display_events.append(record)
    payload=json.dumps({"events":display_events,"images":images, "route":route, "selected":selected, "colors":COLORS, "tiles":tiles, "source":source}, allow_nan=False).replace("<", r"\u003c")
    # Use DOM textContent for all record values; never interpolate them into popup HTML.
    script = r"""
const data=PAYLOAD;
const map=L.map('map',{scrollWheelZoom:false}).setView([13.35,74.79],16);
if(data.tiles){
 const tiles=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);
 tiles.on('tileerror',()=>document.getElementById('notice').textContent='Street tiles unavailable. Coordinates and markers still work; use the offline route view below.');
}
L.control.scale({imperial:false}).addTo(map);
const bounds=[];
let focusedMarker=null;
let segment=[];
function drawSegment(){if(segment.length>1)L.polyline(segment,{color:'#246d94',weight:3,dashArray:'6 6'}).addTo(map).bindTooltip(data.source==='synthetic'?'Synthetic trajectory':'Imported GPS trajectory');segment=[];}
for(const point of data.route){if(point){segment.push(point);bounds.push(point);}else drawSegment();}drawSegment();
 document.getElementById('notice').textContent=(data.source==='synthetic'?'SIMULATED locations':data.source==='mixed'?'MIXED sources — inspect each report':'IMPORTED coordinates — not independently verified')+' · Click pins for reports. Numbers count observations, not severity.';
const groups=new Map();
for(const e of data.events){const key=e.latitude+','+e.longitude;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(e);}
for(const group of groups.values()){
 const e=group.find(e=>e.id===data.selected)||group[0];
 const color=data.colors[e.status]||'#cc6b13';
 const icon=L.divIcon({className:'report-marker',html:'<span style="background:'+color+'"><b>'+group.length+'</b></span>',iconSize:[30,38],iconAnchor:[15,38],popupAnchor:[0,-35]});
 const marker=L.marker([e.latitude,e.longitude],{icon,riseOnHover:true,title:group.length+' report(s) at '+e.latitude+', '+e.longitude}).addTo(map);
 const panel=document.createElement('div');panel.className='report-popup';
 for(const report of group){
  const heading=document.createElement('strong');heading.textContent='Report '+report.id.slice(0,8);panel.appendChild(heading);
  const category=document.createElement('div');category.textContent='Predicted: '+(report.prediction||'unknown').replaceAll('_',' ');panel.appendChild(category);
  if(report.sample_image_key && data.images[report.sample_image_key]){
   const photo=document.createElement('img');photo.src=data.images[report.sample_image_key];photo.alt='Sample '+(report.prediction||'road event')+' photo — not captured at this event';photo.className='sample-photo';panel.appendChild(photo);
   const caption=document.createElement('div');caption.className='sample-caption';caption.textContent='Sample image · category illustration, not event evidence';panel.appendChild(caption);
  }else{const note=document.createElement('div');note.textContent='No sample image available for this category.';panel.appendChild(note);}
  for(const [label,value] of [['Status',report.status],['Location',Number(report.latitude).toFixed(6)+', '+Number(report.longitude).toFixed(6)],['Trip time',Number(report.timestamp).toFixed(2)+' s'],['Speed',report.speed_kmh==null?'Unavailable':Number(report.speed_kmh).toFixed(1)+' km/h'],['Model',report.method],['Probability',report.confidence==null?'Not provided':(report.confidence*100).toFixed(1)+'% (uncalibrated)'],['Source',report.source]]){const line=document.createElement('div');line.textContent=label+': '+value;panel.appendChild(line);}
  panel.appendChild(document.createElement('hr'));
 }
 marker.bindPopup(panel,{minWidth:240,maxWidth:330,maxHeight:310});
 if(group.some(e=>e.id===data.selected))focusedMarker=marker;
 bounds.push([e.latitude,e.longitude]);
}
let laidOut=false;
function layoutMap(){
 const box=document.getElementById('map').getBoundingClientRect();
 if(!box.width || !box.height)return;
 map.invalidateSize();
 if(!laidOut){
  if(bounds.length)map.fitBounds(bounds,{padding:[35,35],maxZoom:19});
  if(focusedMarker){map.setView(focusedMarker.getLatLng(),19);focusedMarker.openPopup();}
  laidOut=true;
 }
}
new ResizeObserver(layoutMap).observe(document.getElementById('map'));
setTimeout(layoutMap,150);
""".replace("PAYLOAD",payload)
    return '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>'+ (vendor/"leaflet.css").read_text()+"""
body{margin:0;font:14px system-ui;color:#172b4d;background:#fff}#map{height:440px;background:#edf2f7}#notice{padding:10px;font-size:12px}.report-marker span{display:flex;width:30px;height:30px;border-radius:50% 50% 50% 0;transform:rotate(-45deg);align-items:center;justify-content:center;color:white;border:2px solid white;box-shadow:0 2px 5px #555;font-weight:bold}.sample-photo{display:block;width:100%;max-height:155px;object-fit:contain;background:#eef1f4;margin:8px 0 4px;border-radius:4px}.sample-caption{font-size:11px;color:#665c45;margin-bottom:8px}.report-marker b{transform:rotate(45deg)}.report-popup{font:13px/1.6 system-ui;max-height:290px;overflow-y:auto}
</style></head><body><div id="map" aria-label="Road report map"></div><div id="notice">SIMULATED coordinates · Click a numbered pin for reports. Pin number = observations at that exact coordinate, not severity.</div><script>"""+(vendor/"leaflet.js").read_text()+"</script><script>"+script+"</script></body></html>"
