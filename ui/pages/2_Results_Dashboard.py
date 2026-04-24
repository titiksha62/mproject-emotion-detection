import streamlit as st
import plotly.express as px

st.title("📊 Results Dashboard")

if "results" not in st.session_state:
    st.warning("⚠️ No results yet. Run analysis first.")
else:
    df = st.session_state["results"]

    top = df.iloc[0]
    second = df.iloc[1]

    # Smart message
    if top["Confidence"] - second["Confidence"] < 15:
        st.warning(f"🤔 Blend: {top['Emotion']} & {second['Emotion']}")
    else:
        st.success(f"🎯 Primary: {top['Emotion']} ({top['Confidence']:.1f}%)")

    # Chart
    fig = px.bar(
        df,
        x="Confidence",
        y="Emotion",
        orientation="h",
        text=df["Confidence"].apply(lambda x: f"{x:.1f}%")
    )

    st.plotly_chart(fig, use_container_width=True)