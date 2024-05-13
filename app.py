import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from my_module import kmeans_predict,kmeans_train,splitTrain,plot_confusion_matrix,evaluate_ratings_full


# Load the Parquet file
@st.cache(ttl=24*3600)
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
    
    df_plot = pd.DataFrame({'Item ID': index, 'Rated Frequency': rate_freq, 'Average Rating': ave_rate.round(2)})
    
    dot_plot = alt.Chart(df_plot).mark_circle().encode(
        x=alt.X('Rated Frequency'),
        y=alt.Y('Average Rating', scale=alt.Scale(domain=[4, 5])),
        size= alt.value(200),  # Circle size proportional to rated frequency
        tooltip=['Item ID', 'Rated Frequency', 'Average Rating'],  # Add additional information to the tooltip
        color=alt.Color('Average Rating', scale=alt.Scale(scheme='viridis')),
    ).properties(
        width=800,
        height=400,
        title='Average Rating and Rated Frequency of Items'  # Title for the graph
    ).interactive()
    
    return dot_plot, df_plot.head(5)  # Return the dot plot and top 10 items DataFrame

# Function to generate dot plot (users)
def generate_dot_plot_user(df, top_x, user_filter='freq'):
    
    if user_filter == 'Rated Frequency':
        df_filtered = df.astype(bool).sum(axis=1).nlargest(top_x)
        index = df_filtered.index
        rate_freq = df_filtered.values
        ave_rate = df.loc[index].replace(0, np.NaN).mean(axis=1).values.round(2)  # Set to None as it's not needed for frequency filter
    elif user_filter == 'Total Ratings':
        df_filtered = df.sum(axis=1).nlargest(top_x)
        index = df_filtered.index
        rate_freq = df.loc[index].astype(bool).sum(axis=1).values
        ave_rate = df.loc[index].replace(0, np.NaN).mean(axis=1).values.round(2)
    
    df_plot = pd.DataFrame({'User ID': index, 'Rated Frequency': rate_freq, 'Average Rating': ave_rate.round(2)})
    
    dot_plot = alt.Chart(df_plot).mark_circle().encode(
        x=alt.X('Rated Frequency'),
        y=alt.Y('Average Rating', scale=alt.Scale(domain=[4, 5])),
        size=alt.value(200),  # Circle size proportional to rated frequency
        tooltip=['User ID', 'Rated Frequency', 'Average Rating'],  # Add additional information to the tooltip
        color=alt.Color('Average Rating', scale=alt.Scale(scheme='plasma')),
    ).properties(
        width=800,
        height=400,
        title='User Average Rating and Frequency'  # Title for the graph
    ).interactive()
    
    return dot_plot, df_plot.head(5)  # Return the dot plot and top 10 items DataFrame


def basic_stats(df):
    num_users=len(df.index)
    num_items=len(df.columns)
    num_user_rates=df.astype(bool).sum().sum()
    sparsity=((((num_users*num_items)-num_user_rates)/(num_users*num_items))*100).round(2)
    ratings=df.values.flatten()
    ratings = ratings[ratings != 0]
    average_rating=ratings.mean().round(2)
    return num_users,num_items,num_user_rates,sparsity,average_rating
    
# Function to plot histogram of ratings
# Function to plot histogram of ratings
def plot_rating_histogram(df):
    ratings = df.values.flatten()
    # Exclude zero ratings
    ratings = ratings[ratings != 0]
    # Plot histogram
    fig = px.histogram(x=ratings, nbins=10)
    fig.update_layout(
        title='Distribution of User Ratings',
        title_x=0.4,
        xaxis_title='User Ratings'
    )
    st.plotly_chart(fig, use_container_width=True)

def load_default_data():
    # Load default data or any other default data you have
    pass

def load_custom_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    
# Define function for evaluation with loading status
def evaluate_with_loading(ui_test, y_pred):
    with st.spinner('Evaluating...'):
      
        TP, TN, FP, FN, precision, recall, f_measure, rmse, mae = evaluate_ratings_full(ui_test, y_pred)
    st.success('Evaluation Completed!')
    return TP, TN, FP, FN, precision, recall, f_measure, rmse, mae

# Main function
def main():
    st.title('Prometheus')
    st.sidebar.title('Navigation')
    data_type = st.sidebar.selectbox('Data Source', ['Default Data', 'Upload Data'])
    
        # Load data based on selection
    if data_type == 'Default Data':
        data = load_data()
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV. Data must be in user-item rating matrix form with items without ratings replaced by 0. The dataset should not have any index", type=['csv'])
        if uploaded_file is not None:
            data = load_custom_data(uploaded_file)
        else:
            data = load_data()
    
    
    tabs = ['Exploratory Data Analysis', 'Model Performance', 'Recommendation Demonstration']
    selected_tab = st.sidebar.radio('Go to', tabs)
    
    
    if selected_tab == 'Exploratory Data Analysis':
        
        #Side bar
        st.sidebar.subheader('Items Options')
        top_x_items = st.sidebar.slider('Select top X items', 5, 500, 20)
        item_filter = st.sidebar.radio('Select display variable', ['Rated Frequency', 'Total Ratings'])
        st.sidebar.subheader('User Options')
        top_x_user = st.sidebar.slider('Select top X Users', 5, 500, 20)
        user_filter = st.sidebar.radio('Select User display variable', ['Rated Frequency', 'Total Ratings'])
        
        st.header('Exploratory Data Analysis')
        
        st.subheader('Basic Statistics')
        num_users,num_items,num_user_rates,sparsity,average_rating=basic_stats(data)
        # Display basic statistics in a vertical list
        st.write("<div>", unsafe_allow_html=True)
        st.write("<ul>", unsafe_allow_html=True)
        
        st.write("<li style='font-size: large;'>Number of Users: <span style='font-size: larger; color: blue;'>{:,}</span></li>".format(num_users), unsafe_allow_html=True)
        st.write("<li style='font-size: large;'>Number of Items: <span style='font-size: larger; color: blue;'>{:,}</span></li>".format(num_items), unsafe_allow_html=True)
        st.write("<li style='font-size: large;'>Number of User-Item Ratings: <span style='font-size: larger; color: blue;'>{:,}</span></li>".format(num_user_rates), unsafe_allow_html=True)
        st.write("<li style='font-size: large;'>Average Rating: <span style='font-size: larger; color: blue;'>{:.2f}</span></li>".format(average_rating), unsafe_allow_html=True)
        st.write("<li style='font-size: large;'>Data Sparsity: <span style='font-size: larger; color: blue;'>{:.2f}%</span></li>".format(sparsity), unsafe_allow_html=True)
        
        st.write("</ul>", unsafe_allow_html=True)
        st.write("</div>", unsafe_allow_html=True)
        st.write(plot_rating_histogram(data))
        
        st.subheader('Analysis of Items')
        dot_plot, top_items_df = generate_dot_plot(data, top_x_items, item_filter)
        st.write(dot_plot)
        
        st.subheader('Top 5 Items')
        st.write(top_items_df)
        
        
        st.subheader('Analysis of Users')

        dot_plot_user, top_users_df = generate_dot_plot_user(data, top_x_user, user_filter)
        st.write(dot_plot_user)
        
        st.subheader('Top 5 Users')
        st.write(top_users_df)

    elif selected_tab == 'Model Performance':
        st.header('Model Performance')
        st.subheader('Algorithm Experimentations')
        subsetRatio = st.slider('Data Subset ratio- Needed due to Streamlit memory limits. 0.1= 10%" of original data used', 0.1, 1.0, 0.2)
        test = st.slider('Test Data Ratio', 0.1, 0.9, 0.2)
        _,dataSubset=splitTrain(data,subsetRatio)
        clustering_algorithm = st.selectbox('Choose Clustering Algorithm', ['Kmeans', 'Ordered Clustering', 'Agglomerative Hierarchical Clustering', 'Gaussian Mixture Model'])
        k_value = st.slider('Select K (Number of clusters)', 1, 40, 5)
        if st.button('Build Model'):
            if clustering_algorithm == 'Kmeans':
                st.write('You selected Kmeans')
                # Add K selection slider
                ui_train,ui_test=splitTrain(dataSubset,test)
                kmeans=kmeans_train(ui_train,k_value)
                st.success('Training Completed!')
                y_pred=kmeans_predict(kmeans,ui_train,ui_test)
                TP, TN, FP, FN, precision, recall, f_measure, rmse, mae = evaluate_with_loading(ui_test, y_pred)
                eval=pd.DataFrame({
                    'Metric':['Precision','Recall','F-Measure','RMSE','MAE'],
                    'Value': [precision, recall, f_measure, rmse, mae]
                })
                st.write('Evaluation Results')
                
                bar_chart = alt.Chart(eval).mark_bar().encode(
                    x=alt.X('Metric:N', title='Metric', axis=alt.Axis(labelAngle=0)),
                    y='Value:Q',
                    text=alt.Text('Value:Q', format='.2f')
                )
                bar_chart_with_text = bar_chart.mark_text(align='center',baseline='bottom',).encode(text='Value:Q')
                st.altair_chart(bar_chart + bar_chart_with_text, use_container_width=True)
                
                st.write(plot_confusion_matrix(TP, TN, FP, FN))
                

            elif clustering_algorithm == 'Ordered Clustering':
                st.write('You selected Ordered Clustering')
                # Add K selection slider
                

            elif clustering_algorithm == 'Agglomerative Hierarchical Clustering':
                st.write('You selected Agglomerative Hierarchical Clustering')
                # Add K selection slider
                

            elif clustering_algorithm == 'Gaussian Mixture Model':
                st.write('You selected Gaussian Mixture Model')
                # Add K selection slider
                

        
        
        
        
        
        
        

    elif selected_tab == 'Recommendation Demonstration':
        st.header('Recommendation Demonstration')
        # Add recommendation demonstration content here

if __name__ == "__main__":
    main()
