import streamlit as st
def show():
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("rect","polygon")
    )

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    # covert hex to rgb
    stroke_color = tuple(int(stroke_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    # realtime_update = st.sidebar.checkbox("Update in realtime", True)
    skip_frame = st.sidebar.slider("Skip frame", 1, 30, 1)
    runtime_type = st.sidebar.selectbox("Runtime type", ("CPU", "GPU"))
    return drawing_mode, stroke_width, stroke_color , skip_frame,runtime_type