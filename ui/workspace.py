"""Presentation helpers for the trip workspace."""
import altair as alt
import pandas as pd
import streamlit as st


def signal_chart(frame, channels=("az",), height=260):
    data=frame.iloc[::max(1,len(frame)//2000)][["timestamp",*channels]].melt("timestamp",var_name="Channel",value_name="Value")
    chart=alt.Chart(data).mark_line(strokeWidth=1.5).encode(
        x=alt.X("timestamp:Q",title="Elapsed time · seconds"),
        y=alt.Y("Value:Q",title="Acceleration · m/s²",scale=alt.Scale(zero=False)),
        color=alt.Color("Channel:N",scale=alt.Scale(range=["#27624f","#487ba0","#aa7337"]),legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("timestamp:Q",format=".2f"),"Channel",alt.Tooltip("Value:Q",format=".3f")]).properties(height=height)
    st.altair_chart(chart,use_container_width=True)


def route_points(frame):
    # Preserve breaks instead of drawing a made-up path across missing fixes.
    points=[]
    for lat,lon in frame.iloc[::50][["latitude","longitude"]].itertuples(index=False,name=None):
        point=[float(lat),float(lon)] if pd.notna(lat) and pd.notna(lon) else None
        if not points or point!=points[-1]:points.append(point)
    return points


def value(number, suffix=""):
    return "Unavailable" if number is None or pd.isna(number) else f"{number:.2f}{suffix}"
