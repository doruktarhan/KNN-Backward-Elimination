import pandas as pd
import numpy as np
import time

#normalize data
def normalize_data(data):
    centralized_data =(data - np.mean(data,axis=0)) / np.std(data,axis=0)
    return centralized_data

def euclidean_distance(rowtest,rowtrain):
    distance = np.linalg.norm(rowtest - rowtrain)
    return distance

def get_neighbours(train_data, test_row):
    distances = []
    for i, train_row in train_data.iterrows():
        distance = euclidean_distance(train_row,test_row)
        distances.append((distance,i))
    distances.sort(key = lambda tup: tup[0])
    neighbours = distances[0:9]
    return neighbours
    

def get_predictions(test_data,data_train_normalized):
    train_data = data_train_normalized
    prediction_sum_list = []
    for i , test_row in test_data.iterrows():
        neighbors = get_neighbours(train_data,test_row)
        test_sum = 0
        for tup in neighbors:
            index = tup[1]
            if data_train_y[index][1] == 1:
                test_sum = test_sum + 1
        prediction_sum_list.append(test_sum)
            
    for i in range(len(prediction_sum_list)):
        if prediction_sum_list[i] > 4.5:
            prediction_sum_list[i] = 1
        else:
            prediction_sum_list[i] = 0
    return prediction_sum_list

def predict(data_test_normalized,data_train_normalized,data_test_y,data_train_y):
    test_labels = data_test_y[:,1]
    predictions = get_predictions(data_test_normalized,data_train_normalized)
    accuracy = 0
    false_positive = 0
    true_positive = 0
    false_negative = 0
    true_negative = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predictions[i]:
            accuracy = accuracy + 1
            if test_labels[i] == 0:
                true_negative = true_negative + 1
            else :
                true_positive = true_positive + 1
    
        else:
            if test_labels[i] == 0:
                false_positive = false_positive + 1
            else :
                false_negative = false_negative + 1
            
                
                
    accuracy = accuracy / len(test_labels)
    
    
    print("TP = ",true_positive, "FN = ", false_negative)
    print("FP = ",false_positive, "TN = ", true_negative)
    
    return accuracy
    

def dropFeature(data_train_normalized,data_test_normalized,data_train_y,data_test_y):
    features_list = data_train_normalized.columns.tolist()
    acc_list = []
    i = 0
    for feature in features_list:
        test_dropped = data_test_normalized.drop([feature],axis = 1)
        train_dropped = data_train_normalized.drop([feature],axis = 1)
        acc_dropped = predict(test_dropped,train_dropped,data_test_y,data_train_y)
        acc_list.append((acc_dropped,feature))
        
    acc_list.sort(key = lambda tup: tup[0],reverse = True)   
    dropped_feature = acc_list[0][1]
    print(dropped_feature, " is the dropped feature")
    acc_final = acc_list[0][0] 
    train_final = data_train_normalized.drop([dropped_feature],axis = 1)
    test_final =  data_test_normalized.drop([dropped_feature],axis = 1)
    return acc_final, train_final, test_final
    
data_train = pd.read_csv("diabetes_train_features.csv")
data_train_normalized = normalize_data(data_train)
data_train_normalized = data_train_normalized.drop(['Unnamed: 0'],axis = 1)


data_test = pd.read_csv("diabetes_test_features.csv")
data_test_normalized = normalize_data(data_test)
data_test_normalized = data_test_normalized.drop(['Unnamed: 0'],axis = 1)

data_train_y = pd.read_csv("diabetes_train_labels.csv").to_numpy()
data_test_y = pd.read_csv("diabetes_test_labels.csv").to_numpy()

time1 = time.time()
accuracy = predict(data_test_normalized,data_train_normalized,data_test_y,data_train_y)


time2 = time.time()
#For 8 features
accuracy7, train_data7, test_data7 = dropFeature(data_train_normalized,data_test_normalized,data_train_y,data_test_y)


time3 = time.time()
#For 7 features
accuracy6, train_data6, test_data6 = dropFeature(train_data7,test_data7,data_train_y,data_test_y)

time4 = time.time()

accuracy5, train_data5, test_data5 = dropFeature(train_data6,test_data6,data_train_y,data_test_y)

time5 = time.time()

accuracy4, train_data4, test_data4 = dropFeature(train_data5,test_data5,data_train_y,data_test_y)
        
time6 =time.time()

print("accuracy wiith all features: ", accuracy)        
print(time2-time1 ," time for 8 feature training")

print("accuracy after first feature dropped: ", accuracy7)
print(time3-time2 ," time for 8 featue 8 times")

print("accuracy after second feature dropped: ", accuracy6)
print(time4-time3, " time for 7 feature 7 times")

print("accuracy after third feature dropped: ", accuracy5)
print(time5-time4, " time for 6 feature 6 times")   

print("accuracy after fourth feature dropped: ", accuracy4)
print(time6-time5, " time for 5 feature 5 times")   
        