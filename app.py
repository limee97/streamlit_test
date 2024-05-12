import streamlit as st
import pandas as pd
import altair as alt

# Load the Parquet file
@st.cache
def load_data():
    return pd.read_parquet('user_item_general.parquet')

# Function to generate top X items plot
def plot_top_items(df, top_x, selection):
    if selection == 'Most Frequently Rated':
        top_items = df.sum().sort_values(ascending=False).head(top_x).reset_index()
        title = 'Most Frequently Rated Items'
    else:
        df_filtered = df[df > 0]  # Exclude ratings of 0
        top_items = df_filtered.mean().sort_values(ascending=False).head(top_x).reset_index()
        title = 'Most Highly Rated Items'
    chart = alt.Chart(top_items, width=600).mark_bar().encode(
        x=alt.X('index', title='Item ID'),
        y=alt.Y(top_items.columns[1], title=title),
        tooltip=['index', top_items.columns[1]]
    ).properties(
        title=title
    )
    st.altair_chart(chart, use_container_width=True)

# Function to generate top X users plot
def plot_top_users(df, top_x, selection):
    if selection == 'Most Rated':
        top_users = df.sum(axis=1).sort_values(ascending=False).head(top_x).reset_index()
        title = 'Top Users with Most Ratings'
    else:
        df_filtered = df[df > 0]  # Exclude ratings of 0
        top_users = df_filtered.mean(axis=1).sort_values(ascending=False).head(top_x).reset_index()
        title = 'Top Users with Highest Average Ratings'
    chart = alt.Chart(top_users, width=600).mark_bar().encode(
        x=alt.X('index', title='User ID'),
        y=alt.Y(top_users.columns[1], title=title),
        tooltip=['index', top_users.columns[1]]
    ).properties(
        title=title
    )
    st.altair_chart(chart, use_container_width=True)

# Main function
def main():
    st.title('Prometheus')
    st.sidebar.title('Navigation')
    tabs = ['Exploratory Data Analysis', 'Model Performance', 'Recommendation Demonstration']
    selected_tab = st.sidebar.radio('Go to', tabs)
    
    if selected_tab == 'Exploratory Data Analysis':
        st.header('Exploratory Data Analysis')
        st.subheader('Items')
        st.sidebar.subheader('Items Options')
        top_x_items = st.sidebar.slider('Select top X items', 1, 20, 5)
        items_selection = st.sidebar.radio('Select items by', ['Most Frequently Rated', 'Most Highly Rated'])
        
        data = load_data()
        plot_top_items(data, top_x_items, items_selection)
        
        st.subheader('Users')
        st.sidebar.subheader('Users Options')
        top_x_users = st.sidebar.slider('Select top X users', 1, 20, 5)
        users_selection = st.sidebar.radio('Select users by', ['Most Rated', 'Highest Average Rated'])
        
        plot_top_users(data, top_x_users, users_selection)

    elif selected_tab == 'Model Performance':
        st.header('Model Performance')
        # Add model performance content here

    elif selected_tab == 'Recommendation Demonstration':
        st.header('Recommendation Demonstration')
        # Add recommendation demonstration content here

if __name__ == "__main__":
    main()
