import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import streamlit as st

#functions
def splitTrain(user_item,test=0.2):
    ui_train, ui_test = train_test_split(user_item, test_size=test, random_state=13)
    return ui_train,ui_test
    
#evaluate all measures
def evaluate_ratings_full(original_ratings, predicted_ratings):
    predicted_ratings = pd.DataFrame(predicted_ratings, index=original_ratings.index, columns=original_ratings.columns)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    squared_error_sum = 0
    absolute_error_sum = 0
    num_ratings = 0

    # Iterate through each rating in the test set
    for index in original_ratings.index:
        for column in original_ratings.columns:
            actual_rating = original_ratings.loc[index, column]
            predicted_rating = predicted_ratings.loc[index, column]

            # Count ratings only if actual rating is not 0
            if actual_rating > 0:
                num_ratings += 1

                # Calculate confusion matrix components
                if actual_rating > 3:
                    if predicted_rating > 3:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if predicted_rating > 3:
                        FP += 1
                    else:
                        TN += 1

                # Calculate squared error for RMSE
                squared_error = (predicted_rating - actual_rating) ** 2
                squared_error_sum += squared_error

                # Calculate absolute error for MAE
                absolute_error = np.abs(predicted_rating - actual_rating)
                absolute_error_sum += absolute_error

    # Calculate evaluation metrics
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_measure = (2 * precision * recall) / (precision + recall)

    # Calculate RMSE and MAE
    rmse = np.sqrt(squared_error_sum / num_ratings)
    mae = absolute_error_sum / num_ratings

    return TP, TN, FP, FN, precision, recall, f_measure, rmse, mae

#Define General Evaluation Function (fmeasure only)
def f_measure(original_ratings, predicted_ratings):
    predicted_ratings=pd.DataFrame(predicted_ratings,index=ui_test.index,columns=ui_test.columns)
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Iterate through each row index in the test set
    for index in original_ratings.index:
        for column in original_ratings.columns:
            actual_rating = original_ratings.loc[index, column]
            predicted_rating = predicted_ratings.loc[index, column]
            
            #only calculate accuracy if actual rating is not 0
            if actual_rating>0:

                if actual_rating > 3:
                    if predicted_rating > 3:
                        TP += 1
                    else:
                        FN += 1
                else:
                
                    if predicted_rating > 3:
                        FP += 1
                    else:
                        TN += 1

    # Calculate evaluation metrics
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_measure = (2 * precision * recall) / (precision + recall)


    return f_measure

def plot_confusion_matrix(TP, TN, FP, FN):
    # Create a DataFrame for the confusion matrix
    confusion_matrix = pd.DataFrame({
        'Predicted Positive': [TP, FP],
        'Predicted Negative': [FN, TN]
    }, index=['Actual Positive', 'Actual Negative'])

    # Plot the confusion matrix as a heatmap
    fig=plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(confusion_matrix, cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(heatmap)
    
    # Add annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion_matrix.iloc[i, j]), ha='center', va='center', color='black')
    
    # Set axis labels and title
    plt.xticks(range(2), confusion_matrix.columns)
    plt.yticks(range(2), confusion_matrix.index)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # Show plot
    st.pyplot(fig)




def kmeans_train(ui_train,k):
    kmeans=KMeans(n_clusters=k,init='k-means++',n_init='auto',algorithm='lloyd',random_state=13).fit(ui_train)
    return kmeans
    
def kmeans_predict(kmeans,ui_train,ui_test):
    predicted_labels=kmeans.predict(ui_test)
    predicted_center_dist=kmeans.transform(ui_test)
    center_dist_matrix=kmeans.transform(ui_train)
    #iterate through each user
    
    ui_test=np.array(ui_test)
    ui_train=np.array(ui_train)
    
    
    #create mask so that only users who rated the item is used for weightage division
    mask=np.where(ui_train !=0,1,ui_train)
    y_pred=np.empty((ui_test.shape[0], ui_test.shape[1]))
    for index,label in enumerate(predicted_labels):
        #create a weightage array, consist of the probability of each user (ui_train) consisting in that cluster. matrix Shape: [1 x users]
        label_rel_dist=abs(predicted_center_dist[index,label]-center_dist_matrix[:,label])
        label_rel_dist=np.where(label_rel_dist==0,0.1,label_rel_dist)  #convert 0 to 0.1 to avoid division by 0. dist of 0 would have meant identical user
        weights=1/label_rel_dist
        
        
        factor=np.dot(mask.T,weights)
        factor=np.where(factor==0,1,factor)  #convert 0 to 1 to avoid division by 0. Won't be a problem since numerator will be 0 as well
    
        predicted_rating=np.transpose(np.dot(ui_train.T,weights)/factor)
        #store predicted rating in y_pred
        y_pred[index]=predicted_rating
    y_pred=np.round(y_pred)
    return y_pred

def oc_train(ui_train,k,sim):
    sim_train=np.corrcoef(np.array(ui_train))
    
    if sim=='cosine':
        #try using cosine similarity
        user_item_matrix=np.array(ui_train)
        dot_product = np.dot(user_item_matrix, user_item_matrix.T)

        # Compute the magnitude of each row (user) in the user-item matrix
        magnitude = np.sqrt(np.sum(user_item_matrix**2, axis=1))

        # Reshape the magnitude array to make it compatible for broadcasting
        magnitude = magnitude.reshape(-1, 1)

        # Compute the cosine similarity matrix
        sim_train = dot_product / (magnitude * magnitude.T)
    
    
    #make diagonals 0
    np.fill_diagonal(sim_train, 0)
    # Round up to 2 decimal point
    sim_train = pd.DataFrame(sim_train, index=ui_train.index, columns=ui_train.index).round(1)

    #form OC
    clusters = {}
    
    if k is not None:
        # Compute the histogram with bin centers
        hist, bin_edges = np.histogram(np.array(sim_train)[np.array(sim_train)>0], bins=k, range=(np.array(sim_train)[np.array(sim_train)>0].min(), np.array(sim_train)[np.array(sim_train)>0].max()))
        cluster_center = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute the centers of the bins
        
        #create the clusters based on average:
        for cluster_key in cluster_center:
            clusters[cluster_key] = set()
            
        temp=clusters.copy()
        
        for i, row in sim_train.iterrows():
            for j, sim_value in row.items():
                if sim_value > 0:
                    # Find the cluster Ck where k = SMij
                    closest_key = min(clusters.keys(), key=lambda x: abs(x - sim_value))
                    # Insert users i and j into cluster Ck
                    clusters[closest_key].add(i)
                    clusters[closest_key].add(j)
        
    else:
            # Iterate through each entry in the similarity matrix
        for i, row in sim_train.iterrows():
            for j, sim_value in row.items():
                if sim_value > 0:
                    # Find the cluster Ck where k = SMij
                    cluster_key = sim_value
                
                     # If the cluster Ck is not found, create it
                    if cluster_key not in clusters:
                        clusters[cluster_key] = set()
                    
                    # Insert users i and j into cluster Ck
                    clusters[cluster_key].add(i)
                    clusters[cluster_key].add(j)
        # Sort clusters by k in descending order
    
    oc_cluster = dict(sorted(clusters.items(), key=lambda item: item[0], reverse=True))
    

    return oc_cluster

def oc_predict(oc_cluster,ui_train,ui_test,sim):
    #form similarity matrix for test data
    test_users=ui_test.index
    ui=pd.concat([ui_train,ui_test])
    sim_test=np.corrcoef(np.array(ui))
    
    if sim=='cosine':
        #try using cosine similarity
        user_item_matrix=np.array(ui)
        dot_product = np.dot(user_item_matrix, user_item_matrix.T)

        # Compute the magnitude of each row (user) in the user-item matrix
        magnitude = np.sqrt(np.sum(user_item_matrix**2, axis=1))

        # Reshape the magnitude array to make it compatible for broadcasting
        magnitude = magnitude.reshape(-1, 1)

        # Compute the cosine similarity matrix
        sim_test = dot_product / (magnitude * magnitude.T)
    
    
    
    #remove train dataset
    np.fill_diagonal(sim_test, 0)
    sim_test=sim_test[ui_train.shape[0]:]
    sim_test = pd.DataFrame(sim_test, index=ui_test.index, columns=ui.index).round(2)
    
    #assign user to cluster:
    max_sim=np.array(sim_test.max(axis=1))
    predicted_labels= np.array([min(oc_cluster.keys(), key=lambda x: abs(x - value)) for value in max_sim])

    ui_test=np.array(ui_test)
    ui_train=np.array(ui_train)
    
    # Initialize an empty DataFrame to store the predicted ratings
    y_pred=np.empty((ui_test.shape[0], ui_test.shape[1]))
    
    #predict for all test users
    
    for row,label in enumerate(predicted_labels):
        #get all users of that cluster
        cluster_users=np.array(list(oc_cluster[label]))
        weights=np.empty(cluster_users.shape[0])
        #print('empty weight shape:',weights.shape)
        #get a list of weights for all cluster users
        #print(label,cluster_users)
        for i,user in enumerate(cluster_users):
            weights[i]=sim_test.loc[test_users[row],user]
        
        #convert all negative simialrity to 0 as they are dissimilar
        weights[weights<0]=0
        cluster_user_ratings=ui.loc[cluster_users]
        #print(row)
        #print(label)
        
        #create mask to avoid using weights users that never rated an item
        mask=np.where(cluster_user_ratings !=0,1,cluster_user_ratings)
        division_factor=np.dot(weights,mask)
        division_factor=np.where(division_factor==0,1,division_factor)   # convert those with 0 weightage to 1 as the numerator would be 0 anyway
        predicted_ratings=(np.dot(weights,np.array(cluster_user_ratings)))/(division_factor)
        y_pred[row]=predicted_ratings
        #print("Completed Test Iter:",row)
        
    y_pred=np.round(y_pred)
    
    return y_pred 

def ahc_predict(ui_train,ui_test,k):
    ui=pd.concat([ui_train,ui_test])
    #create model and cluster using full data. Unable to fit for training data only as scikilearn ahc does not support direct prediction of new data
    ahc = AgglomerativeClustering(n_clusters=k,metric='l2',linkage='average').fit(np.array(ui))
    #generate clustered labels
    predicted_labels=ahc.labels_
    
    #slice to only contain labels for ui_train users
    predicted_labels_train=predicted_labels[:ui_train.shape[0]]
    #slice to only contain labels for ui_test users
    predicted_labels_test=predicted_labels[ui_train.shape[0]:]

    
    #iterate through each user
    ui_test=np.array(ui_test)
    ui_train=np.array(ui_train)
    
    
    #create empty np to store predicted values for ui_test
    y_pred=np.empty((ui_test.shape[0], ui_test.shape[1]))
    
    #create weightage matrix (based on distance)
    dist_matrix=euclidean_distances(np.array(ui))
    dist_matrix=np.where(dist_matrix==0,0.1,dist_matrix) #convert distances that are 0 to 0.1 (very small) since those points are almost identical to the test user
    weight_matrix=1/dist_matrix
    #slice weight matrix to only contain rows from ui_test and cols from ui_train
    weight_matrix=weight_matrix[ui_train.shape[0]:,:ui_train.shape[0]]
    
    
    #iterate through each ui_test users
    for index,label in enumerate(predicted_labels_test):
        #create a weightage array, consist of the probability of each user (ui_train) consisting in that cluster. matrix Shape: [1 x users]
    
        #filter for users in same cluster as test user
        users_in_k = [index for index, value in enumerate(predicted_labels_train) if value == label]
        ratings=ui_train[users_in_k]
        weights=weight_matrix[index,users_in_k]
        #create mask so that only users who rated the item is used for weightage division
        mask=np.where(ratings !=0,1,ratings)
        
        factor=np.dot(mask.T,weights)
        factor=np.where(factor==0,1,factor)  #convert 0 to 1 to avoid division by 0. Won't be a problem since numerator will be 0 as well
    
        predicted_rating=np.transpose(np.dot(ratings.T,weights)/factor)
        #store predicted rating in y_pred
        y_pred[index]=predicted_rating
    y_pred=np.round(y_pred)
    return y_pred

def gmm_predict(ui_train,ui_test,k):
    ui_train=np.array(ui_train)
    ui_test=np.array(ui_test)
    
    #training
    gmm=GaussianMixture(k, covariance_type='diag', random_state=42).fit(ui_train)
    #probabilities, each col is a cluster each row is a user
    prob=gmm.predict_proba(ui_train)

    
    #predict
    predicted_labels=gmm.predict(ui_test)
    #create empty array to store results
    y_pred=np.empty((ui_test.shape[0], ui_test.shape[1]))
    #create mask so that only users who rated the item is used for weightage division
    mask=np.where(ui_train !=0,1,ui_train)
    
    #iterate through each user
    for index,label in enumerate(predicted_labels):
    #create a weightage array, consist of the probability of each user (ui_train) consisting in that cluster. matrix Shape: [1 x users]
        weights=prob[:,label]
        factor=np.dot(mask.T,weights)
        factor=np.where(factor==0,1,factor)  #convert 0 to 1 to avoid division by 0. Won't be a problem since numerator will be 0 as well
    
        predicted_rating=np.transpose(np.dot(ui_train.T,weights)/factor)
        #store predicted rating in y_pred
        y_pred[index]=predicted_rating

    return y_pred
