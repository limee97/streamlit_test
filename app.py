import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Parquet file
@st.cache
def load_data():
    return pd.read_parquet('user_item_general.parquet')

# Function to generate top X items plot
def plot_top_items(df, top_x, selection):
    if selection == 'Most Frequently Rated':
        top_items = df.sum().nlargest(top_x)
        title = 'Most Frequently Rated Items'
    else:
        top_items = df.mean().nlargest(top_x)
        title = 'Most Highly Rated Items'
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_items.index, y=top_items.values)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Item ID')
    plt.ylabel('Rating Frequency' if selection == 'Most Frequently Rated' else 'Average Rating')
    plt.title(title)
    st.pyplot()

# Function to generate top X users plot
def plot_top_users(df, top_x, selection):
    if selection == 'Most Rated':
        top_users = df.sum(axis=1).nlargest(top_x)
        title = 'Top Users with Most Ratings'
    else:
        top_users = df.mean(axis=1).nlargest(top_x)
        title = 'Top Users with Highest Average Ratings'
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_users.values, y=top_users.index)
    plt.xlabel('Rating Frequency' if selection == 'Most Rated' else 'Average Rating')
    plt.ylabel('User ID')
    plt.title(title)
    st.pyplot()

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
