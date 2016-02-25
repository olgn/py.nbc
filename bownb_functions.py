# -*- coding: utf-8 -*-
#This file contains the functions used by bownb.py to perform binary and 
#trinary naive-bayes classification. The code was written by Teal
#Hobson-Lowther. 

import glob
import numpy as np 
import math as mth

#This function preprocesses the data found in data_folder, and uses p*100% as
# training and (1-p)*100% as testing:
def bow_train(data_folder,p):
    
    data_str = data_folder + "/*" #Set the data string
    genre = glob.glob(data_str) #Get all children of data string
    file_directory, bags_train,bags_test = [],[],[] #Some empty variables
    
    #Cycle through each topic:
    for genre_str in genre:
        print("Reading %s" %genre_str)
        print("...")
        print('\n')
        
        #Grab all the documents from a genre:
        topic_str = genre_str + "\*"
        file_directory = glob.glob(topic_str)
        
        text_train, text_test, test = "","",[] #Some empty variables
        
        num_docs = len(file_directory) #number of documents in the directory
        train_docs = mth.floor(num_docs*p) #number to keep for training
       
        for file in file_directory[0:train_docs]:
                #open each training document and append the contents after header
                #to an array:
                intl = open(file).read()
                without_header = intl.partition("Lines:")[2]
                text_train += " " + without_header
    
        for file in file_directory[train_docs:num_docs]:
            #open each testing document and append the contents after header
            #to an array:
            intl = open(file).read()
            text_test = intl.partition("Lines:")[2]
            text_test = clean_text(text_test)
            test.append(text_test) 
        
        bags_test.append(test)
        text_train = clean_text(text_train)
        bags_train.append(text_train)
    
    # Now bags_train and bags_test have all the words together by topic
    
    # Create a bag that has probabilities of each word associated with it:
    prob_bags_train = []
    for bag in bags_train:
        unique, counts = np.unique(bag,return_counts = True)
        counts = counts/len(bag)
        prob_bag = []
        prob_bag.append(unique)
        prob_bag.append(counts)
        prob_bags_train.append(prob_bag)
    
    return (bags_train,bags_test,prob_bags_train)

# This converts a topic to a class number:
def topic2class(topic_str):
    return {
        'Atheism': 0,
        'Graphics': 1,
        'MS-Windows': 2,
        'PC Hardware': 3,
        'Mac Hardware' : 4,
        'Windows 10' : 5,
        'For Sale': 6,
        'Autos': 7,
        'Motorcycles': 8 ,
        'Baseball': 9,
        'Hockey': 10,
        'Cryptology': 11,
        'Electronics': 12,
        'Medical': 13,
        'Space': 14,
        'Christianity': 15,
        'Guns': 16,
        'Middle East': 17,
        'Politics': 18,
        'Religion': 19
    }[topic_str]
    
#This converts a whole text collection to a matrix of all words in that document. 
def clean_text(text):
    #List of bad symbols to remove:
    bad_symbols = [":",".","\n","/","@", "\\","*","=","^",";","_","|",
                               '"',"' "," '","-",
                               "(",")",",",">","<",
                               "!","?","[","]","+",
                               "&","%","$","#","~","{","}"]
                               
    #Cycle through and remove bad symbols:
    for symbol in bad_symbols:
        text = text.replace(symbol," ")
    
    #Split the text collection into individual words:
    text= text.lower().split()
    
    #Replace digits:
    text = [x for x in text if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
    
    return text

# This performs binary classification via a naive bayes method:
def bow_binary(topic_1,topic_2,training_probs,test_files):
    print("Comparing %s and %s:"%(topic_1, topic_2))
    
    # Find the class id's of class 1 and class 2:
    class_1,class_2 = topic2class(topic_1),topic2class(topic_2)
    
    #Get the testing data associated with each of the two classes:
    c1_test,c2_test = test_files[class_1],test_files[class_2]
    
    #Get the probability bags found during training for each class:
    probs_1, probs_2 = training_probs[class_1],training_probs[class_2]
    
    #Calculate the probability adjustment for laplacian smoothing:
    adj1 = 1/(len(probs_1[0]))
    adj2 = 1/(len(probs_2[0]))
    
    #Some empty variables:
    labels, cc1_labels,cc2_labels = [],[],[]
    
    # Calculate priors for each test data using class_calc:
    c1_data,c2_data = class_calc(c1_test),class_calc(c2_test)
    l = len(c2_data)
    
    # Find log probability of the priors, p(class1), p(class2):
    p1 = mth.log(len(probs_1[0])/(len(probs_1[0]) + len(probs_2[0])))
    p2 = mth.log(len(probs_2[0])/(len(probs_1[0]) + len(probs_2[0])))
    
    # Cycle through test data and classify:
    print("Classifying %s Data..." %topic_1)
    for file in c1_data:
        pc1 = p1
        pc2 = p2
        # Add the log probability of each document given class 1 or 2, using
        # the prior and the output of log_prob.
        # This calculates the log probability of each class:
        pc1 = pc1 + log_prob(file[0],probs_1, adj1)
        pc2 = pc2 + log_prob(file[0],probs_2, adj2)
        
        # If p(Class 1) > p(Class 2), append a "0" to cc1_labels:
        # Else, append a "1". 
        if max(pc1,pc2) == pc1:
            cc1_labels.append(0)
        else:
            cc1_labels.append(1)
            
    print("Classifying %s Data..." %topic_2)
    for file in c2_data:
        pc1 = p1
        pc2 = p2
        # Add the log probability of each document given class 1 or 2, using
        # the prior and the output of log_prob.
        # This calculates the log probability of each class:
        pc1 = pc1 + log_prob(file[0],probs_1, adj1)
        pc2 = pc2 + log_prob(file[0],probs_2, adj2)
        
        # If p(Class 1) > p(Class 2), append a "0" to cc2_labels:
        # Else, append a "1". 
        if max(pc1,pc2) == pc1:
            cc2_labels.append(0)
        else:
            cc2_labels.append(1)
        
    #Generate a confusion matrix based on the results of the previous
    #classification:
    C = [[l-sum(cc1_labels),sum(cc1_labels)],[l-sum(cc2_labels),sum(cc2_labels)]] 
    
    #Put the  cc1_labels and cc2_labels arrays into an output matrix, labels:
    labels.append(cc1_labels)
    labels.append(cc2_labels)
    
    #Calculate the accuracy of the classification using the confusion matrix C:
    accuracy = 1-(C[1][0] + C[0][1])/np.sum(C)
    
    #Display the results of the classification:
    print("Confusion Matrix:")
    print("%s vs. %s"%(topic_1,topic_2))
    print(C)
    print("Overall classification accuracy: %.3f." %accuracy)
    
    #Return the labels, confusion matrix, and overall accuracy:
    return (labels,C, accuracy)
        
# This perfoms trinary classification via a Naive Bayes Method:
def bow_trinary(topic_1,topic_2,topic_3,training_probs,test_files):
    
    # Find the class id's of class 1, class 2, and class 3:
    class_1,class_2,class_3 = topic2class(topic_1),topic2class(topic_2),topic2class(topic_3)
    
    #Get the testing data associated with each of the three classes:
    c1_test,c2_test,c3_test = test_files[class_1],test_files[class_2],test_files[class_3]
    
    #Get the probability bags found during training for each class:
    probs_1, probs_2, probs_3 = training_probs[class_1],training_probs[class_2],training_probs[class_3]
    
    #Calculate the probability adjustment for laplacian smoothing:
    adj1,adj2,adj3 = 1/(len(probs_1[0])),1/(len(probs_2[0])),1/(len(probs_2[0]))
    
    # Create some empty variables:
    labels, cc1_labels,cc2_labels,cc3_labels = [],[],[],[]
    
    # Calculate word probabilities for each test data:
    c1_data,c2_data,c3_data = class_calc(c1_test),class_calc(c2_test),class_calc(c3_test)
    
    # Find log probability of the priors, p(class1), p(class2), and p(class3)
    # based on all words found during training:
    s = len(probs_1[0]) + len(probs_2[0]) + len(probs_3[0])
    p1 = mth.log(len(probs_1[0])/s)
    p2 = mth.log(len(probs_2[0])/s)
    p3 = mth.log(len(probs_3[0])/s)
    
    # Cycle through test data and see what it gets classified as:
    print("Classifying %s Data..." %topic_1)
    for file in c1_data:
        
        pc1 = p1
        pc2 = p2
        pc3 = p3
        
        # Add the log probability of each document given class 1, 2, or 3 using
        # the prior and the output of log_prob.
        # This calculates the log probability of each class:
        pc1 = pc1 + log_prob(file[0],probs_1, adj1)
        pc2 = pc2 + log_prob(file[0],probs_2, adj2)
        pc3 = pc3 + log_prob(file[0],probs_3, adj3)
        
        #If the maximum class probability calculation above indicates
        #that the correct class is Class 1, output a label "0". If Class 2
        # is the maximum, output a "1". If Class 3 is the most likely, output
        # the label "2".
        if max(pc1,pc2,pc3) == pc1:
            cc1_labels.append(0)
        if max(pc1,pc2,pc3) == pc2:
            cc1_labels.append(1)
        if max(pc1,pc2,pc3) == pc3:
            cc1_labels.append(2)
            
    print("Classifying %s Data..." %topic_2)
    for file in c2_data:
        pc1 = p1
        pc2 = p2
        pc3 = p3
        
        # Add the log probability of each document given class 1, 2, or 3 using
        # the prior and the output of log_prob.
        # This calculates the log probability of each class:
        pc1 = pc1 + log_prob(file[0],probs_1, adj1)
        pc2 = pc2 + log_prob(file[0],probs_2, adj2)
        pc3 = pc3 + log_prob(file[0],probs_3, adj3)

        #If the maximum class probability calculation above indicates
        #that the correct class is Class 1, output a label "0". If Class 2
        # is the maximum, output a "1". If Class 3 is the most likely, output
        # the label "2".
        if max(pc1,pc2,pc3) == pc1:
            cc2_labels.append(0)
        if max(pc1,pc2,pc3) == pc2:
            cc2_labels.append(1)
        if max(pc1,pc2,pc3) == pc3:
            cc2_labels.append(2)
    
    print("Classifying %s Data..." %topic_3)
    for file in c3_data:
        pc1 = p1
        pc2 = p2
        pc3 = p3
        
        # Add the log probability of each document given class 1, 2, or 3 using
        # the prior and the output of log_prob.
        # This calculates the log probability of each class:
        pc1 = pc1 + log_prob(file[0],probs_1, adj1)
        pc2 = pc2 + log_prob(file[0],probs_2, adj2)
        pc3 = pc3 + log_prob(file[0],probs_3, adj3)
        
        #If the maximum class probability calculation above indicates
        #that the correct class is Class 1, output a label "0". If Class 2
        # is the maximum, output a "1". If Class 3 is the most likely, output
        # the label "2".
        if max(pc1,pc2,pc3) == pc1:
            cc3_labels.append(0)
        if max(pc1,pc2,pc3) == pc2:
            cc3_labels.append(1)
        if max(pc1,pc2,pc3) == pc3:
            cc3_labels.append(2)
        
    #Generate a confusion matrix based on the results of the classification
    #performed above:
    C = [[cc1_labels.count(0),cc1_labels.count(1),cc1_labels.count(2)],
          [cc2_labels.count(0),cc2_labels.count(1),cc2_labels.count(2)],
           [cc3_labels.count(0), cc3_labels.count(1), cc3_labels.count(2)]]
           
    #Put the  cc1_labels, cc2_labels, and cc3_labels
    #arrays into an output matrix, labels:
    labels.append(cc1_labels)
    labels.append(cc2_labels)
    labels.append(cc3_labels)
    
    #Calculate overall classification accuracy using the confusion matrix, C:
    accuracy = 1-(C[1][0] + C[2][0] + C[0][1] + C[0][2] + C[1][2] + C[2][1])/np.sum(C)
    
    #Summarize results of trinary classification:
    print("Confusion Matrix:")
    print("%s vs. %s vs. %s"%(topic_1,topic_2,topic_3))
    print(C)
    print("Overall classification accuracy: %.2f" %accuracy)
    
    #Return the labels, confusion matrix, and overall accuracy:
    return (labels,C, accuracy)

#This function calculates the sum log probability of a document given
#a specific class.
def log_prob(words, prob_mat, adj):
    log_probs = []
    
    #Calculate log of the adjustment probability for laplacian smoothing:
    cnst = mth.log(adj)
    
    #Cycle through all words and calculate the log probabilities for each:
    for w in words:
        if (w in prob_mat[0]):
            log_probs.append(mth.log(prob_mat[1][prob_mat[0] == w] + adj))
        else:
            log_probs.append(cnst)
            
    #Return the sum of all of these log likelihoods:
    return sum(log_probs)
    
#This function turns a matrix of test documents test_class into an array with
# the probability of each word contained in a document: 
def class_calc(test_class):
    c_data = []
    
    #Cycle through documents and calculate probabiltiy of each word in that
    #document:
    for test_doc in test_class:
        unique, counts = np.unique(test_doc,return_counts = True)
        counts = counts/len(test_doc)
        prob_bag = []
        prob_bag.append(unique)
        prob_bag.append(counts)
        c_data.append(prob_bag)  
    return c_data