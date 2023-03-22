import fileinput
import os
import numpy as np
import math
from random import shuffle

#  Initialize parameters and labels
L = ['e','j','s']
S = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']

theta_e = [0] * 27
theta_j = [0] * 27
theta_s = [0] * 27
likelihoodList = [0]*27

K_s = len(S)
K_l = len(L)

# bag = [0] * 27

conf_matrix = np.array([[0,0,0],[0,0,0],[0,0,0]])

# Method to calculate the estimated conditional probability
def cond_prob(alpha, a_s, c_k):

    # Set probability counter variables
    numerator = 0
    denominator = 0

    # Go through each file in the training set 
    for filename in os.listdir("languageID"):
        with open(os.path.join("languageID", filename), 'r') as f:

            # For each language
            for y in L:
                # from 0.txt to 9.txt
                for train in range(10):
                    if(filename.endswith(y+ str(train)+".txt")):

                        while(True):
                            # Read each character
                            x = f.read(1)

                            # Skip new lines
                            if(x=="\n"):
                                continue
                               
                            # At the end of file   
                            if(x==''):
                                break

                            else:
                                
                                # Update probability counters
                                if(x==a_s and y==c_k):
                                    numerator+=1
                                if((x in S) and y==c_k):
                                    denominator+=1

                        

                        break      

    # Update numerator and denominator
    # print(numerator, denominator, a_s)
    numerator+=alpha
    denominator+=(K_s*alpha)
    # print((numerator/denominator))

    return (numerator/denominator)

def bagofwords(file):

    bag = [0] * 27

    while(True):
    
        x = file.read(1)

        # Skip new lines
        if(x=="\n"):
            continue
                               
        # At the end of file   
        if(x==''):
            break      
        
        if(x == " "):
            bag[26]+=1
        else:
            bag[ord(x)-97] +=1
        
    # print(bag)
    return bag

def likelihood(x, y):
    
    likelihoodVal = 0
    likelihoodFinal = 1

    for i in range(K_s):
        if(y == "e"):
            likelihoodVal += (x[i]* np.log(theta_e[i]))
        if(y == "j"):
            likelihoodVal += (x[i]* np.log(theta_j[i]))
        if(y == "s"):
            likelihoodVal += (x[i]* np.log(theta_s[i]))

    return likelihoodVal

    if(y == "e"):
        for i in range(K_s-1):
            
            likelihoodVal += (x[i]* np.log(theta_e[i]))
            
            # print(theta_e[i], bag[i])
            # print(likelihoodVal)
        # print('e:', likelihoodVal)
        likelihoodFinal = math.e**likelihoodVal
        # likelihoodFinal = np.log(-likelihoodVal)
        # print(likelihoodFinal)
        return likelihoodVal
    if(y == "j"):
        for i in range(K_s):
            likelihoodVal += (x[i]* np.log(theta_j[i]))
            
            # print(theta_e[i], bag[i])
            # print(likelihoodVal)
        # print('j:', likelihoodVal)
        likelihoodFinal = math.e**likelihoodVal
        # print(likelihoodFinal)
        return likelihoodVal
    if(y == "s"):
        for i in range(K_s):
            likelihoodVal += (x[i]* np.log(theta_s[i]))
            # likelihoodList[i] = x[i]* np.log(theta_s[i])
            # print(theta_e[i], bag[i])
            # print(likelihoodVal)
        # print('s:', likelihoodVal)
        
        likelihoodFinal = math.e**likelihoodVal
        # print(likelihoodFinal)   
        return likelihoodVal     
    
def prior(alpha, c_k):
    val = 0
    for filename in os.listdir("languageID"):
        with open(os.path.join("languageID", filename), 'r') as f:
            
            # For each language
            for y in L:
            
                for train in range(10):
                        if(filename.endswith(y+ str(train)+".txt")):
                            if(filename[0] == c_k):
                                val+=1
    val += alpha
    val = val/(30+K_l*alpha)
    # print(val)
    return val

def naive_bayes(x, y):
    return likelihood(x,y) + np.log(prior(0.5, y))
    

def predict(x):

    # print(naive_bayes(x,'e'), naive_bayes(x,'j'), naive_bayes(x,'s'))

    if(naive_bayes(x,'e') > naive_bayes(x,'j')):
        if(naive_bayes(x,'e') > naive_bayes(x,'s')):
            return 'e'
        else:
            return 's'
    else:
        if(naive_bayes(x,'j') > naive_bayes(x,'s')):
            return 'j'
        else:
            return 's'


# CODE START ************************************************************************************************
for i in range(K_s):
    # print(S[i])
    theta_e[i] = cond_prob(0.5,S[i],'e')
    theta_j[i] = cond_prob(0.5,S[i],'j')
    theta_s[i] = cond_prob(0.5,S[i],'s')


# Q2-1 PRINT PRIOR PROBABILITIES  *******************************************************************************
# print(prior(0.5, 'e'))
# print(prior(0.5, 'j'))
# print(prior(0.5, 's'))

# Q2-2 & Q2-3 PRINT CLASS CONDITIONAL PROBABILITIES ********************************************************************
# print(theta_e)
# print(theta_j)
# print(theta_s)


# Q2-4 PRINT BAG OF WORDS VECTOR x *************************************************************************************
# x = bagofwords(file1)

# Q2-5 PRINT LIKELIHOODs *********************************************************************************************
# file1 = open("languageID/e10.txt")
# x = bagofwords(file1)
# file1.close()
# print(likelihood(x,'e'))
# print(likelihood(x,'j'))
# print(likelihood(x,'s'))

# Q2-6 PRINT POSTERIOR PROBABILITIES *************************************************************************************
# file1 = open("languageID/e10.txt")
# x = bagofwords(file1)
# file1.close()
# print(naive_bayes(x,'e'))
# print(naive_bayes(x,'j'))
# print(naive_bayes(x,'s'))
# print("Prediction:", predict(x))



# Q2-7 Predict using 10.txt to 19.txt in each language as testing set ********************************************************
# for filename in os.listdir("languageID"):
#         with open(os.path.join("languageID", filename), 'r') as f:
            
#             # For each language
#             for y in L:
            
#                 for train in range(10,20):
#                         if(filename.endswith(y+ str(train)+".txt")):
#                             x = bagofwords(f)
#                             # print(filename, predict(x), y, naive_bayes(x,'e'), naive_bayes(x,'j'), naive_bayes(x,'s'))
#                             if(y == 'e'):
#                                 if(predict(x)=='e'):
#                                     conf_matrix[0][0]+=1
#                                 if(predict(x)=='s'):
#                                     conf_matrix[1][0]+=1
#                                 if(predict(x)=='j'):
#                                     conf_matrix[2][0]+=1
#                             if(y == 's'):
#                                 if(predict(x)=='e'):
#                                     conf_matrix[0][1]+=1
#                                 if(predict(x)=='s'):
#                                     conf_matrix[1][1]+=1
#                                 if(predict(x)=='j'):
#                                     conf_matrix[2][1]+=1
#                             if(y == 'j'):
#                                 if(predict(x)=='e'):
#                                     conf_matrix[0][2]+=1
#                                 if(predict(x)=='s'):
#                                     conf_matrix[1][2]+=1
#                                 if(predict(x)=='j'):
#                                     conf_matrix[2][2]+=1        
# # Print the resulting confusion matrix
# print(conf_matrix)


# Q2-8 Shuffle order of characters in test document (s10.txt) and evaluate the output ***************************************


# First show output of normal file
file1 = open("languageID/s10.txt")
x = bagofwords(file1)
print("Normal File")
print("Bag of characters: ", x)
print("Prediction: ", predict(x))
print()

file1 = open("languageID/s10.txt")
input = list(file1.read())
shuffle(input)
scrambled = ''.join(input)

# Write new file of shuffled text
with open('shuffled.txt', 'w') as f:
    f.write(scrambled)
file_shuffle = open("shuffled.txt")

x = bagofwords(file_shuffle)

# Print prediction of shuffled file
print("Shuffled File")
print("Bag of characters: ", x)
print("Prediction: ", predict(x))
