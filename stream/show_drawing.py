    
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

def show(stroke_width, stroke_color, bg_image, drawing_mode, scale_width, scale_height, json_data ,key = "canvas"):   
    ############################################################
    #### Area and counting settings.                       #####
    ############################################################
        
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#eee",
        background_image= Image.fromarray(bg_image) if bg_image is not None else None,
        update_streamlit=True,
        width= 640,
        height= 480,
        drawing_mode=drawing_mode,
        initial_drawing=json_data if json_data is not None else None,
        key=key,
        display_toolbar=True
    )
    

    # st.image(canvas_result.image_data)

    if json_data is not None:
        canvas_result.json_data = json_data

    # Area
    with st.expander("Area", expanded=False):
        with st.form(key=key):
            if canvas_result.json_data is not None:
                for i, k in enumerate([k for k in canvas_result.json_data['objects'] if k['type'] == 'rect']):
                    cols = st.columns([3,3,4])
                    k['area_name'] = (cols[0].text_input('Area ' + str(i), 'Area ' + str(i))) 
                    # k['crowd_levels'] = (cols[1].slider('Choose medium crowd level', 0, 100, (1,50)))
                    k['crowd_levels'] = {}
                    k['crowd_levels']['medium'] = (cols[1].number_input('Medium crowd level', 0, 100, 5))
                    k['crowd_levels']['high'] = (cols[1].number_input('High crowd level', 0, 200, 20))
                    k['crowd_levels']['very_high'] = (cols[1].number_input('Very high crowd level', 0, 300, 30))
                    x1, y1 = scale_width*k['left'], scale_height*k['top']
                    x2, y2 = scale_width*(k['left'] + k['width']), scale_height*(k['top'] + k['height'])
                    cols[2].write([x1, y1, x2,  y2])
                    
            st.form_submit_button("Create areas")
        
        areas_info = {}
        areas_draw = {}
        areas_crowd = {}
        if canvas_result.json_data is not None:
            for i, k in enumerate([k for k in canvas_result.json_data['objects'] if k['type'] == 'rect']):
                x1, y1 = scale_width*k['left'], scale_height*k['top']
                x2, y2 = scale_width*(k['left'] + k['width']), scale_height*(k['top'] + k['height'])
                areas_info[str(k['area_name'])] = [x1, y1, x2,  y2]
                areas_crowd[str(k['area_name'])] = k['crowd_levels']
                areas_draw[str(k['area_name'])] = [k['stroke'], k['strokeWidth']]
                   
    return areas_info, areas_draw, areas_crowd