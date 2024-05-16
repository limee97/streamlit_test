import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import my_module
from my_module import kmeans_predict,kmeans_train,splitTrain,plot_confusion_matrix,evaluate_ratings_full
import time

# Load the Parquet file
@st.cache_resource
def load_data():
    return pd.read_parquet('user_item_general.parquet')

@st.experimental_dialog("User Ratings",width="large")
def show_dialog(item_titles):
    df=get_id(st.session_state.selected,item_titles)
    df['Rating']=3
    st.write("Scroll The Table To The Right To Rate (1-5)")
    st.session_state.userRate=st.data_editor(df,disabled =('Item','title'))
    
    #st.write(st.session_state)
    if st.button("Done"):
        
        if (st.session_state.userRate['Rating'] < 1).any() or (st.session_state.userRate['Rating'] > 5).any():
            st.warning('Invalid Rating, Please ensure values are between 1-5')
        else:
            st.session_state.ratingFilled=True
            st.rerun()

    
def create_rating_table(data):
    ratings = pd.Series(index=data.index, dtype=float)  # Initialize ratings column with NaN
    data['Rating'] = ratings  # Add the ratings column to the DataFrame
    return data

def get_id(titles,df):
    filtered_df = df[df['title'].isin(titles)]
    filtered_df=filtered_df.drop_duplicates('title')
    return filtered_df
# Function to generate dot plot
def generate_dot_plot(df, top_x, item_titles,item_filter='freq'):
    
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
    df_plot=df_plot.merge(item_titles, how='left', left_on='Item ID', right_on='item')
    dot_plot = alt.Chart(df_plot).mark_circle().encode(
        x=alt.X('Rated Frequency'),
        y=alt.Y('Average Rating', scale=alt.Scale(domain=[4, 5])),
        size= alt.value(200),  # Circle size proportional to rated frequency
        tooltip=['Item ID','title', 'Rated Frequency', 'Average Rating'],  # Add additional information to the tooltip
        color=alt.Color('Average Rating', scale=alt.Scale(scheme='viridis')),
    ).properties(
        width=800,
        height=400,
        title='Average Rating and Rated Frequency of Items'  # Title for the graph
    ).interactive()
    
    return dot_plot, df_plot.head(5)  # Return the dot plot and top 10 items DataFrame
# Function to initialize session state
def initialize_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'edited_df' not in st.session_state:
        st.session_state.edited_df = None
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

def load_item_titles():
    return pd.read_parquet('item_titles.parquet')

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
        time.sleep(1)  # Simulating evaluation process
        TP, TN, FP, FN, precision, recall, f_measure, rmse, mae = evaluate_ratings_full(ui_test, y_pred)
    st.success('Evaluation Completed!')
    return TP, TN, FP, FN, precision, recall, f_measure, rmse, mae

def save_edits(edited_df):
    st.session_state["df"]=edited_df



def get_recommend(model,user_rate,user_model,data):
    ui_test = (pd.DataFrame(data.iloc[-1]).T)*0.0
    # Iterate over each row in new_df and fill in user_df with the ratings
    for index, row in user_rate.iterrows():
        item_col = row['item']
        ui_test[item_col] = ui_test[item_col]+row['Rating']
    
    #do prediction
    id=my_module.general_recommend(user_model,model,ui_test,data)
    return id

















# Main function
def main():
    initialize_session_state()
    st.title('Prometheus')
    st.sidebar.title('Navigation')
    data_type = st.sidebar.selectbox('Data Source', ['Default Data', 'Upload Data'])
    
        # Load data based on selection
    if data_type == 'Default Data':
        data = load_data()
        item_titles = load_item_titles()
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV. Data must be in user-item rating matrix form with items without ratings replaced by 0. The dataset should not have any index", type=['csv'])
        uploaded_file_title = st.sidebar.file_uploader("Upload item description csv. The first column should be item ID and second column is description", type=['csv'])
        if uploaded_file is not None:
            data = load_custom_data(uploaded_file)
        else:
            data = load_data()
        if uploaded_file_title is not None:
            item_titles = load_custom_data(uploaded_file_title)
    
    
    tabs = ['Exploratory Data Analysis', 'Model Performance', 'Recommendation Demonstration']
    selected_tab = st.sidebar.radio('Go to', tabs)
    
    
    if selected_tab == 'Exploratory Data Analysis':
        
        #Side bar
        st.sidebar.subheader('Items Options')
        top_x_items = st.sidebar.slider('Select top X items', 5, 50, 20)
        item_filter = st.sidebar.radio('Select display variable', ['Rated Frequency', 'Total Ratings'])
        st.sidebar.subheader('User Options')
        top_x_user = st.sidebar.slider('Select top X Users', 5, 50, 20)
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
        dot_plot, top_items_df = generate_dot_plot(data, top_x_items, item_titles,item_filter)
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
        subsetRatio = st.slider('Data Subset ratio- Needed due to Streamlit memory limits. 0.1= 10%" of original data used', 0.1, 0.99, 0.2)
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
                ).interactive()
                bar_chart_with_text = bar_chart.mark_text(align='center',baseline='bottom',).encode(text='Value:Q')
                st.altair_chart(bar_chart + bar_chart_with_text, use_container_width=True)
                
                st.write(plot_confusion_matrix(TP, TN, FP, FN))
                

            elif clustering_algorithm == 'Ordered Clustering':
                st.write('You selected Ordered Clustering')
                # Add K selection slider
                 # Add K selection slider
                ui_train,ui_test=splitTrain(dataSubset,test)
                oc=my_module.oc_train(ui_train,k_value,sim='pearson')
                st.success('Training Completed!')
                y_pred=my_module.oc_predict(oc,ui_train,ui_test,sim='pearson')
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
                ).interactive()
                bar_chart_with_text = bar_chart.mark_text(align='center',baseline='bottom',).encode(text='Value:Q')
                st.altair_chart(bar_chart + bar_chart_with_text, use_container_width=True)
                
                st.write(plot_confusion_matrix(TP, TN, FP, FN))               

            elif clustering_algorithm == 'Agglomerative Hierarchical Clustering':
                st.write('You selected Agglomerative Hierarchical Clustering')
                ui_train,ui_test=splitTrain(dataSubset,test)
                st.success('Training Completed!')
                y_pred=my_module.ahc_predict(ui_train,ui_test,k_value)
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
                ).interactive()
                bar_chart_with_text = bar_chart.mark_text(align='center',baseline='bottom',).encode(text='Value:Q')
                st.altair_chart(bar_chart + bar_chart_with_text, use_container_width=True)
                
                st.write(plot_confusion_matrix(TP, TN, FP, FN))   
                

            elif clustering_algorithm == 'Gaussian Mixture Model':
                st.write('You selected Gaussian Mixture Model')
                ui_train,ui_test=splitTrain(dataSubset,test)
                st.success('Training Completed!')
                y_pred=my_module.gmm_predict(ui_train,ui_test,k_value)
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
                ).interactive()
                bar_chart_with_text = bar_chart.mark_text(align='center',baseline='bottom',).encode(text='Value:Q')
                st.altair_chart(bar_chart + bar_chart_with_text, use_container_width=True)
                
                st.write(plot_confusion_matrix(TP, TN, FP, FN))   
                

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    elif selected_tab == 'Recommendation Demonstration':
        st.header('Recommendation Demonstration')
        st.subheader('Step 1: Build a Model')
        subsetRatio = st.slider('Data Subset ratio- Needed due to Streamlit memory limits. 0.1= 10%" of original data used', 0.1, 0.99, 0.2)
        _,dataSubset=splitTrain(data,subsetRatio)
        dataSubset_titles=item_titles[item_titles['item'].isin(dataSubset.columns)]
        st.session_state.clustering_algorithm = st.selectbox('Choose Clustering Algorithm', ['Kmeans', 'Ordered Clustering', 'Agglomerative Hierarchical Clustering', 'Gaussian Mixture Model'])
        k_value = st.slider('Select K (Number of clusters)', 1, 40, 5)
        if st.button('Build Model'):
            if st.session_state.clustering_algorithm == 'Kmeans':
                st.write('You selected Kmeans')
                # Add K selection slider
                st.session_state.user_model=kmeans_train(dataSubset,k_value)
                st.success('Training Completed!')
            

                

            elif st.session_state.clustering_algorithm == 'Ordered Clustering':
                st.write('You selected Ordered Clustering')
                # Add K selection slider
                 # Add K selection slider
                st.session_state.user_model=my_module.oc_train(dataSubset,k_value,sim='pearson')
                st.success('Training Completed!')
       

            elif st.session_state.clustering_algorithm == 'Agglomerative Hierarchical Clustering':
                st.write('You selected Agglomerative Hierarchical Clustering')
                st.session_state.user_model=my_module.ahc_model_only(k_value)
                st.success('Training Completed!')

                

            elif st.session_state.clustering_algorithm == 'Gaussian Mixture Model':
                st.write('You selected Gaussian Mixture Model')
                st.session_state.user_model=my_module.gmm_train(dataSubset,k_value)
                st.success('Training Completed!')
            
            st.session_state.BuildModel=True
    
        # Display interface for selecting items with dropdown
        if 'BuildModel' in st.session_state:
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.subheader('Step 2: Rate Some Items')
            selected_item_titles=0
            selected_item_titles = st.multiselect('Select items to rate', dataSubset_titles['title'].tolist())
            st.session_state.selected=selected_item_titles 

            if st.button('Confirm Item Selection'):
                show_dialog(item_titles)
        
            if 'ratingFilled' in st.session_state:
                st.success('Items Rated!')
                st.text("")
                st.text("")
                st.text("")
                st.write('Your Ratings:')
                st.dataframe(st.session_state.userRate)
                st.text("")
                st.text("")
                st.text("")
                st.subheader('Step 3: Show your recommended items!')
                st.write('Model Used: ',st.session_state.clustering_algorithm)
                st.write('Number of Clusters: ',k_value)
                st.text("")
                #st.write(st.session_state)
                if st.button('Get Recommendations'):
                    id=get_recommend(st.session_state.clustering_algorithm,st.session_state.userRate,st.session_state.user_model,dataSubset)
                    top_10 = pd.DataFrame({'Item_ID': id})
                    top_10 = top_10.merge(item_titles, left_on='Item_ID', right_on='item', how='left').drop(columns='item')
                    top_10.index = top_10.index + 1
                    #st.write(id)
                    st.success('Recommendations Made!')
                    st.write('Top 10 Recommended Items for You:')
                    st.dataframe(top_10)
                    
                    
                
                
                
    
        

        

            




if __name__ == "__main__":
    main()
