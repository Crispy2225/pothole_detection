"""Variable vehicle-like signals for simulation, not a calibrated dynamics model."""
import json
import numpy as np
import pandas as pd
from config import DATA, CLASSES, SAMPLE_RATE, WINDOW_SECONDS


def generate_trip(trip_id, seed, windows=60):
    rng=np.random.default_rng(seed)
    n=windows*SAMPLE_RATE*WINDOW_SECONDS
    t=np.arange(n)/SAMPLE_RATE
    # Continuous speed with acceleration, cruising and braking phases.
    knots=np.arange(0,len(t),500)
    targets=rng.uniform(12,48,len(knots))
    speed=np.interp(np.arange(n),knots,targets)+.4*np.sin(t*.4)
    speed=np.clip(speed,3,55)
    heading=.5+.18*np.sin(t/20)+.08*np.sin(t/7)
    meters=speed/3.6/SAMPLE_RATE
    north=np.cumsum(meters*np.cos(heading));east=np.cumsum(meters*np.sin(heading))
    lat=13.3500+north/111320;lon=74.7900+east/(111320*np.cos(np.deg2rad(13.35)))
    # GPS fixes held for one second, with meter-scale error and occasional loss.
    fix=np.arange(n)//SAMPLE_RATE
    fixes=int(fix.max())+1
    lat=lat[(fix*SAMPLE_RATE)]+rng.normal(0,1.5/111320,fixes)[fix]
    lon=lon[(fix*SAMPLE_RATE)]+rng.normal(0,1.5/108000,fixes)[fix]
    lost=rng.random(fixes)<.025
    lat[lost[fix]]=np.nan;lon[lost[fix]]=np.nan
    suspension=rng.uniform(.7,1.4);mounting=rng.uniform(.8,1.15)
    white=rng.normal(0,rng.uniform(.08,.22),n)
    correlated=np.convolve(white,[.2,.6,.2],mode="same")
    az=correlated+.10*np.sin(2*np.pi*(8+speed/15)*t)+.08*np.sin(t*.6)
    labels=rng.choice(CLASSES,windows,p=[.67,.12,.09,.12])
    for w,label in enumerate(labels):
        begin=w*100;end=begin+100;u=np.arange(100)/SAMPLE_RATE
        center=rng.uniform(.45,1.35);dt=u-center
        scale=mounting*(.6+.4*speed[begin:end]/30)*rng.uniform(.55,1.6)
        if label=="pothole":
            width=rng.uniform(.035,.11)*suspension
            pulse=-rng.uniform(2.5,6)*np.exp(-(dt/width)**2)+rng.uniform(1.2,4)*np.exp(-((dt-.13)/(.10*suspension))**2)
            pulse+=np.where(dt>0,.7*np.exp(-np.maximum(dt,0)/.35)*np.sin(2*np.pi*6*dt),0)
        elif label=="speed_breaker":
            width=rng.uniform(.12,.32)*suspension
            pulse=rng.uniform(1.5,4.5)*np.exp(-(dt/width)**2)-rng.uniform(.8,2.8)*np.exp(-((dt-.30)/width)**2)
        elif label=="rough_road":
            pulse=rng.uniform(.3,1.2)*np.sin(2*np.pi*rng.uniform(5,12)*u)+rng.normal(0,.45,100)
        else:
            pulse=np.zeros(100)
            if rng.random()<.20: # Road joints are negative examples, sometimes sharp.
                pulse=rng.uniform(.5,2.7)*np.exp(-(dt/rng.uniform(.03,.10))**2)
        az[begin:end]+=scale*pulse
    ax=np.gradient(speed/3.6)*SAMPLE_RATE+rng.normal(0,.08,n)
    yaw=np.gradient(heading)*SAMPLE_RATE
    ay=(speed/3.6)*yaw+rng.normal(0,.08,n)
    gyro=.025*np.gradient(az)*SAMPLE_RATE/10
    return pd.DataFrame(dict(trip_id=trip_id,window_id=np.arange(n)//100,timestamp=t,
        ax=ax,ay=ay,az=az,gx=gyro+rng.normal(0,.025,n),gy=.6*gyro+rng.normal(0,.02,n),gz=yaw+rng.normal(0,.015,n),
        speed_kmh=speed,latitude=lat,longitude=lon,label=np.repeat(labels,100)))


def main():
    directory=DATA/"scenarios";directory.mkdir(parents=True,exist_ok=True)
    manifest={"source":"synthetic","generator_version":2,"seed":2026,"sample_rate_hz":50,"window_seconds":2,
        "description":"Continuous speed, acceleration/braking/turning, imbalanced randomized classes, variable impact/rebound, suspension variation, road joints, 1 Hz GPS with noise and loss. Not a validated vehicle model; path does not follow surveyed streets.",
        "units":{"acceleration":"m/s², gravity removed","gyro":"rad/s","speed":"km/h"},
        "splits":{"train":list(range(12)),"validation":list(range(12,16)),"test":list(range(16,20)),"demo":[20]}}
    for i in range(21):generate_trip(i,2026+i).to_csv(directory/f"trip_{i:02d}.csv",index=False)
    (DATA/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("Generated 21 version-2 synthetic trips, 60 windows each.")

if __name__=="__main__":main()
