# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 23:20:34 2015

@author: Teal
"""

# Requires glob, math as mth, numpy as np:
def pre_process_nb(data_folder):
    
    data_str = data_folder + "/*"
    genre = glob.glob(data_str)
    topics, file_directory, bags_train,bags_test = [],[],[],[]
    # Create a list of all of the genres and get the resulting file directories
    for genre_str in genre:
        
        print("Reading in %s" %genre_str)
        print('\n')
        
        #Grab all the files from a genre:
        topic_str = genre_str + "\*"
        file_directory = glob.glob(topic_str)
        text_train, text_test = "",""
        
        #Process the text. For each genre, we keep p% of the values for testing.
        p = .8
        num_docs = len(file_directory)
        train_docs = mth.floor(num_docs*p)
        bad_symbols = [":",".","\n","/","@",
                               '"',"' "," '","-",
                               "(",")",",",">","<",
                               "!","?","[","]","+",
                               "&","%","$","#","~"]
        for file in file_directory[0:train_docs]:
                text_train += " " + open(file).read()
    
        for file in file_directory[train_docs:num_docs]:
            text_test += " " + open(file).read()
            
        for symbol in bad_symbols:
            text_train = text_train.replace(symbol," ")
            text_test = text_test.replace(symbol," ")
        text_train = text_train.lower().split()
        text_test = text_test.lower().split()
        text_train = [x for x in text_train if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
        text_test = [x for x in text_test if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
        bags_train.append(text_train)
        bags_test.append(text_test)
    
    # Now bags has all the words together by topic
    # i) Atheism ii) Graphics iii) MS-Windows iv) PC Hardware v) Mac Hardware 
    # vi) Windows 10 vii) For Sale viii) Autos ix) Motorcycles x) Baseball
    # xi) Hockey xii) Cryptology xiii) Electronics xiv) Medical xv) Space
    # xvi) Christianity, xvii) Guns xviii) Middle East xix) Politics xx) Religion 
    # Create a bag that has probabilities of each word associated with it:
    prob_bags_train = []
    for bag in bags_train:
        unique, counts = np.unique(bag,return_counts = True)
        counts = counts/len(bag)
        prob_bag = []
        prob_bag.append(unique)
        prob_bag.append(counts)
        prob_bags_train.append(prob_bag)
    
    return prob_bags_train
    
