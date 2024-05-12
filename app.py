import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Load the Parquet file
@st.cache_resource
def load_data():
    return pd.read_parquet('user_item_general.parquet')

# Function to generate dot plot
def generate_dot_plot(df, top_x, item_filter='freq'):
    
    if item_filter == 'Rated Frequency':
        df_filtered = df.astype(bool).sum(axis=0).nlargest(top_x)
        index = df_filtered.index
        rate_freq = df_filtered.values
        ave_rate = df[index].replace(0, np.NaN).mean().round(2)  # Set to None as it's not needed for frequency filter
    elif item_filter == 'Total Ratings':
        df_filtered = df.sum(axis=0).nlargest(top_x)
        index = df_filtered.index
        rate_freq = df[index].astype(bool).sum()
        ave_rate = df[index].replace(0, np.NaN).mean().round(2)
    
    df_plot = pd.DataFrame({'Item ID': index, 'Rated Frequency': rate_freq, 'Average Rating': ave_rate})
    
    dot_plot = alt.Chart(df_plot).mark_circle().encode(
        x=alt.X('Rated Frequency'),
        y=alt.Y('Average Rating', scale=alt.Scale(domain=[4, 5])),
        size='Rated Frequency',  # Circle size proportional to rated frequency
        tooltip=['Item ID'],
        color=alt.Color('Average Rating', scale=alt.Scale(scheme='viridis')),
    ).properties(
        width=700,
        height=500
    ).interactive()
    
    return dot_plot

# Main function
def main():
    st.title('Prometheus')
    st.sidebar.title('Navigation')
    tabs = ['Exploratory Data Analysis', 'Model Performance', 'Recommendation Demonstration']
    selected_tab = st.sidebar.radio('Go to', tabs)
    data = load_data()
    if selected_tab == 'Exploratory Data Analysis':
        st.header('Exploratory Data Analysis')
        st.sidebar.subheader('Items Options')
        top_x_items = st.sidebar.slider('Select top X items', 10, 500, 10)
        item_filter = st.sidebar.radio('Select display variable', ['Rated Frequency', 'Total Ratings'])
        
        
        dot_plot=generate_dot_plot(data, top_x_items,item_filter)
        st.write(dot_plot)

    elif selected_tab == 'Model Performance':
        st.header('Model Performance')
        # Add model performance content here

    elif selected_tab == 'Recommendation Demonstration':
        st.header('Recommendation Demonstration')
        # Add recommendation demonstration content here

if __name__ == "__main__":
    main()
