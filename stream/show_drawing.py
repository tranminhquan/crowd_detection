    
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
    # for i in canvas_result.json_data['objects']:
    #     for k, v in i.items():
    #         print(k, v)
    # Area
    with st.expander("Area", expanded=False):
        with st.form(key=key):
            if canvas_result.json_data is not None:
                for i, k in enumerate([k for k in canvas_result.json_data['objects'] if k['type'] == 'rect']):
                    cols = st.columns([3,3,4])
                    k['area_name'] = (cols[0].text_input('Area ' + str(i), 'Area ' + str(i))) 
                    # k['crowd_levels'] = (cols[1].slider('Choose medium crowd level', 0, 100, (1,50)))
                    k['crowd_levels'] = {}
                    k['crowd_levels']['medium'] = (cols[1].number_input('Medium crowd level', 0, 100, 5,key='medium'+str(i) + key))
                    k['crowd_levels']['high'] = (cols[1].number_input('High crowd level', 0, 200, 20,key='high'+str(i) + key))
                    k['crowd_levels']['very_high'] = (cols[1].number_input('Critical crowd level', 0, 300, 30,key='very_high'+str(i) + key))
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
    lines_info = {}
    lines_draw = {}

    # with st.expander("Queue line", expanded=False):
    #     with st.form(key="Line" + key):
    #         if canvas_result.json_data is not None:
    #             for i, k in enumerate([k for k in canvas_result.json_data['objects'] if k['type'] == 'line']):
    #                 cols = st.columns(2)
    #                 k['line'] = (cols[0].text_input('Line ' + str(i), k['line'] if 'line' in k else 'Line ' + str(i)))                    
    #                 x1, y1 = scale_width*(k['left'] + k['x1']), scale_height*(k['top'] + k['y1'])
    #                 x2, y2 = scale_width*(k['left'] + k['x2']), scale_height*(k['top'] + k['y2'])
    #                 cols[1].write([x1, y1, x2,  y2])
                    
    #         st.form_submit_button("Create lines")
            

    #     lines_info = {}
    #     lines_draw = {}
       
    #     if canvas_result.json_data is not None:
    #         for i, k in enumerate([k for k in canvas_result.json_data['objects'] if k['type'] == 'line']):
    #             x1, y1 = scale_width*(k['left'] + k['x1']), scale_height*(k['top'] + k['y1'])
    #             x2, y2 = scale_width*(k['left'] + k['x2']), scale_height*(k['top'] + k['y2'])
    #             lines_info[str(k['line'])] = [x1, y1, x2,  y2]
    #             # lines_draw[str(k['line'])] = [tuple(int(k['stroke'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)), k['strokeWidth']]     
    
    with st.expander("Queue region", expanded=False):
        with st.form(key="Polygon" + key):
            if canvas_result.json_data is not None:
                for i, k in enumerate([k for k in canvas_result.json_data['objects'] if k['type'] == 'path']):
                    cols = st.columns(2)
                    k['queue'] = (cols[0].text_input('Queue region ' + str(i), k['queue'] if 'queue' in k else 'Queue region ' + str(i)))                    
                    # x1, y1 = scale_width*(k['left'] + k['x1']), scale_height*(k['top'] + k['y1'])
                    # x2, y2 = scale_width*(k['left'] + k['x2']), scale_height*(k['top'] + k['y2'])
                    x1,y1 = k['path'][0][1], k['path'][0][2]
                    x2,y2 = k['path'][1][1], k['path'][1][2]
                    x3,y3 = k['path'][2][1], k['path'][2][2]
                    x4,y4 = k['path'][3][1], k['path'][3][2]
                    cols[1].write([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                    
            st.form_submit_button("Queue region")

        queue_info = {}
        
        if canvas_result.json_data is not None:
            for i, k in enumerate([k for k in canvas_result.json_data['objects'] if k['type'] == 'path']):
                x1,y1 = int(scale_width*k['path'][0][1]), int(scale_height*k['path'][0][2])
                x2,y2 = int(scale_width*k['path'][1][1]), int(scale_height*k['path'][1][2])
                x3,y3 = int(scale_width*k['path'][2][1]), int(scale_height*k['path'][2][2])
                x4,y4 = int(scale_width*k['path'][3][1]), int(scale_height*k['path'][3][2])
                queue_info[str(k['queue'])] = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                # queue_draw[str(k['queue'])] = [tuple(int(k['stroke'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)), k['strokeWidth']]
    return areas_info, areas_draw, areas_crowd,lines_info,queue_info
