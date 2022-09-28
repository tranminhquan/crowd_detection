from pathlib import Path
import plotly.express as px  
import streamlit as st
import pandas as pd

def show(results,frames,placeholder,result_queue):
    df = pd.DataFrame(results)
    #add column time_end = time + 1/fps to calculate the duration of each area
    # df['time_end'] = df['time'] + 1
    # print(df)
    #group by area and time to calculate the duration of each area
    df_ = pd.DataFrame(result_queue)
    df_queue = df_.groupby(['queue', 'time']).agg({'queue_count': 'max'}).reset_index()
    df_waiting = df_.groupby(['queue', 'time']).agg({ 'wait_time': 'mean'}).reset_index()
    df = df.groupby(['area', 'time', 'time_end']).agg({'count': 'max','people_count': 'max', 'crowd_level': 'max'}).reset_index()
    # df = df.groupby(['area', 'time', 'time_end','crowd_level']).agg({'count': 'mean', 'crowd_level': }).reset_index()
    with placeholder.container():
    
        #plot the results
        tab1, tab2, tab3, tab4, tab5= st.tabs(['Count', 'Crowd level','Table','Queue', 'Waiting time'])
        with tab1:
            fig = px.line(df.iloc[-200:], x="time", y="count", color='area',line_shape="linear", render_mode="svg")
            st.write(fig)
        with tab2:
            fig2 = px.timeline(df.iloc[-200:],x_start="time",x_end="time_end", y="area", color="crowd_level", color_discrete_map={'low': 'green', 'medium': 'yellow', 'high': 'orange', 'very_high': 'red'})
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
        for tab,video in zip(frame_tab, list(frames.keys())):
            with tab:
                # get height of frame
                height, width, _ = frames[video].shape
                if height > 500:
                    st.image(frames[video],width=300)
                else:  
                    st.image(frames[video],width=640)  

        st.write(df_waiting)