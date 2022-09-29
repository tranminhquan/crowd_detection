from pathlib import Path
import plotly.express as px  
import streamlit as st
import pandas as pd

def show(results,result_queue,frames,placeholder,areas_info,queue_in_area):
    df = pd.DataFrame(results)
    df_ = pd.DataFrame(result_queue)
    df_queue = df_.groupby(['queue', 'time']).agg({'queue_count': 'max'}).reset_index()
    df_waiting = df_.groupby(['queue', 'time']).agg({ 'wait_time': 'mean'}).reset_index()
    df = df.groupby(['area', 'time', 'time_end']).agg({'count': 'max','people_count': 'max', 'crowd_level': 'max'}).reset_index()
    # df = df.groupby(['area', 'time', 'time_end','crowd_level']).agg({'count': 'mean', 'crowd_level': }).reset_index()
    color_discrete_map={'1low': 'green', '2medium': 'goldenrod', '3high': 'orange', '4critical': 'red'}
    with placeholder.container():
    
        #plot the results
        tab1, tab2, tab3, tab4, tab5= st.tabs(['Count', 'Crowd level','Table','Queue', 'Waiting time'])
        with tab1:
            fig = px.line(df.iloc[-200:], x="time", y="count", color='area',line_shape="linear", render_mode="svg")
            st.write(fig)
        with tab2:
            fig2 = px.timeline(df.iloc[-200:],x_start="time",x_end="time_end", y="area", color="crowd_level", color_discrete_map=color_discrete_map)
            # fig2.update_xaxes(tickformat = '%S')
            st.write(fig2)
        with tab3:
            df_tab = st.tabs(list(df['area'].unique()))
            for tab,area in zip(df_tab, list(df['area'].unique())):
                with tab:
                    st.write(df[df['area'] == area])
        with tab4:
            fig = px.line(df_queue.iloc[-200:], x="time", y="queue_count", color='queue',line_shape="linear", render_mode="svg")
            st.write(fig)
        with tab5:
            fig = px.line(df_waiting.iloc[-200:], x="time", y="wait_time", color='queue',line_shape="linear", render_mode="svg")
            fig.update_yaxes(range=[0, 30])
            st.write(fig)
        # split frame to dynamic tab
        frame_tab = st.tabs(list(map(lambda x: Path(x).stem,frames.keys())))
        df = pd.DataFrame(results)
        df = df.groupby(['area']).agg({'count': 'max','people_count': 'max', 'crowd_level': 'max'}).reset_index()
        for tab,video in zip(frame_tab, list(frames.keys())):
            with tab:
                cols = st.columns(6)
                cols[0].write('Area')
                cols[1].write('Count')
                cols[2].write('Crowd level')
                cols[3].write('Queue')
                cols[4].write('Queue count')
                cols[5].write('Waiting time')
                for i, (area_name, area) in enumerate(areas_info[video].items()):
                    area_name_text = f'<p style="font-family:sans-serif; color:black; font-size: 14px;">{area_name}</p>'
                    cols[0].markdown(area_name_text, unsafe_allow_html=True)
                    count = df[df['area'] == area_name]['count'].values[-1]
                    color = color_discrete_map[df[df['area'] == area_name]['crowd_level'].values[-1]]
                    count_text = f'<p style="font-family:sans-serif; color:{color}; font-size: 14px;">{count}</p>'
                    cols[1].markdown(count_text, unsafe_allow_html=True)
                    crouded_level = df[df['area'] == area_name]['crowd_level'].values[-1]
                    crouded_level_text = f'<p style="font-family:sans-serif; color:{color}; font-size: 14px;">{crouded_level}</p>'
                    cols[2].markdown(crouded_level_text, unsafe_allow_html=True)
                    for queue in queue_in_area[area_name]:
                        queue_text = f'<p style="font-family:sans-serif; color:black; font-size: 14px;">{queue}</p>'
                        cols[3].markdown(queue_text, unsafe_allow_html=True)
                        queue_count = df_queue[df_queue['queue'] == queue]['queue_count'].values[-1]
                        queue_count_text = f'<p style="font-family:sans-serif; color:black; font-size: 14px;">{queue_count}</p>'
                        cols[4].markdown(queue_count_text, unsafe_allow_html=True)
                        wait_time = int(df_waiting[df_waiting['queue'] == queue]['wait_time'].values[-1])
                        wait_time_text = f'<p style="font-family:sans-serif; color:black; font-size: 14px;">{wait_time}</p>'
                        cols[5].markdown(wait_time_text, unsafe_allow_html=True)
                    line_break = f'<p style="font-family:sans-serif; color:white; font-size: 14px;">a</p>'
                    for i in range(len(queue_in_area[area_name]) - 1):
                        cols[0].markdown(line_break, unsafe_allow_html=True)
                        cols[1].markdown(line_break, unsafe_allow_html=True)
                        cols[2].markdown(line_break, unsafe_allow_html=True)
                height, width, _ = frames[video].shape
                if height > 500:
                    st.image(frames[video],width=300)
                else:  
                    st.image(frames[video],width=640)  
