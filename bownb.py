# -*- coding: utf-8 -*-
#This code contains the script for preprocessing some text document data,
#applying a binary naive-bayes classifier to six pairs of document topics, and
#then performing trinary naive-bayes classification on six groups of three
#document classes. The functions found herein are contained in
#bownb_functions.py, which requires packages glob, numpy, and math.
#This code was written by Teal Hobson-Lowther.
import bownb_functions as pp

#List of Topics:
# i) Atheism ii) Graphics iii) MS-Windows iv) PC Hardware v) Mac Hardware 
# vi) Windows 10 vii) For Sale viii) Autos ix) Motorcycles x) Baseball
# xi) Hockey xii) Cryptology xiii) Electronics xiv) Medical xv) Space
# xvi) Christianity, xvii) Guns xviii) Middle East xix) Politics xx) Religion 

# Get a matrix with all of the training words and probabilites:
(train,test,train_probs) = pp.bow_train("newsgroups",.8)
print("Finished processing data. Now performing pairwise binary classification...")

# Test the classifier on several binary combinations of genres: 
(l1,C1,acc1) = pp.bow_binary("Atheism","Cryptology",train_probs, test)
(l2,C2,acc2) = pp.bow_binary("Politics","Guns",train_probs, test)
(l3,C3,acc3) = pp.bow_binary("Christianity","Religion",train_probs, test)
(l4,C4,acc4) = pp.bow_binary("Hockey","Cryptology",train_probs,test)
(l5,C5,acc5) = pp.bow_binary("Electronics","Middle East",train_probs,test)
(l6,C6,acc6) = pp.bow_binary("Middle East","Politics",train_probs,test)
print("Finished binary classification. Now performing trinary classification...")

# Test the classifier on groups of three:
(lt1,Ct1, acct1) = pp.bow_trinary("Atheism","Cryptology","Guns",train_probs, test)
(lt2,Ct2, acct2) = pp.bow_trinary("Hockey","Baseball","Autos",train_probs, test)
(lt3,Ct3, acct3) = pp.bow_trinary("Religion","Medical","Graphics",train_probs, test)
(lt4,Ct4, acct4) = pp.bow_trinary("PC Hardware","Windows 10","MS-Windows",train_probs, test)
(lt5,Ct5, acct5) = pp.bow_trinary("Atheism","Christianity","Religion",train_probs, test)
(lt6,Ct6, acct6) = pp.bow_trinary("Space","Electronics","Guns",train_probs, test)
print("Finished trinary classification. Program complete.")