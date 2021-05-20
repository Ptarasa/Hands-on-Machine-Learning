          
import numpy as np
import re

def Sentence_Probability(Words, Vocab):
    Prob = 0
    Prob_S = 0
    for word in Vocab:
        if word in Words:
            Prob = Prob + np.log(Vocab[word][3])
            Prob_S = Prob_S + np.log(Vocab[word][2])
        else:
            Prob = Prob + np.log((1 - Vocab[word][3]))
            Prob_S = Prob_S + np.log((1 - Vocab[word][2]))
    Prob = np.exp(Prob)
    Prob_S = np.exp(Prob_S)
    return Prob, Prob_S


def predict(Words, Vocab, Spam, Ham):
    PS  = Spam / (Spam + Ham)
    P_S = Ham / (Spam + Ham)
    PslS, Psl_S = Sentence_Probability(Words, Vocab)
    A = np.exp(np.log(PslS) + np.log(PS))
    B = np.exp(np.log(Psl_S) + np.log(P_S))
    probability = 1/(1 + np.exp(np.log(B) - np.log(A)))
    if probability > 0.5:
        return 1
    else:
        return 0


File = open(input("Enter the Traning File name: "), "r", encoding = 'unicode-escape')
Spam = 0
Ham = 0
Vocab = {} # { word : [ham, spam, p(w|-s), p(w|s)] }
for line in File:
    line = line.lower() #lower case
    line = line.strip() # white space
    line = re.sub("[!#?,.:';]", '', line)
    Words = line[2:]
    Words = Words.split() # words
    Words = list(dict.fromkeys(Words)) # duplicate words
    if line[0] == '1':
        Spam = Spam + 1
        for word in Words:
            if word in Vocab:
                Vocab[word][1] = Vocab[word][1] + 1
            else:
                Vocab[word] = [0,1,0,0]
    else:
        Ham = Ham + 1
        for word in Words:
            if word in Vocab:
                Vocab[word][0] = Vocab[word][0] + 1
            else:
                Vocab[word] = [1,0,0,0]

File.close()     

# Vocabulary Created
k = 1

# Setting Probabilities
for Word in Vocab:
    Vocab[Word][2] = (k+Vocab[Word][0])/((2 * k) + Ham)
    Vocab[Word][3] = (k+Vocab[Word][1])/((2 * k) + Spam)
File = open(input("Enter the StopWord file: "),"r", encoding = 'unicode-escape')

for line in File:
    line.lower()
    line.strip()
    line = re.sub("[!#?,.:';]", '', line)
    if line in Vocab:
        Vocab.pop(line)
File = open(input("Enter the Test File: "), "r", encoding = 'unicode-escape')
test_spam = test_ham = 0
TP = FP = TN = FN = 0

for line in File:
    line = line.lower() #lower case
    line = line.strip() # white space
    line = re.sub("[!#?,.:';]", '', line)
    Words = line[2:]
    Words = Words.split() # words
    Words = list(dict.fromkeys(Words)) # duplicate words
    if line[0]=='1':
        test_spam = test_spam + 1
        v = predict(Words, Vocab, Spam, Ham)
        if v == 1:
            TP = TP + 1
        else :
            FP = FP + 1
    else:
        test_ham = test_ham + 1
        v = predict(Words, Vocab, Spam, Ham)
        if v == 0:
            TN = TN + 1
        else:
            FN = FN + 1
            
print(" Ham count = " + str(test_ham))
print(" Spam count = " + str(test_spam))

print(" True Positive =" + str(TP))
print(" False Positive =" + str(FP))
print(" True Negative =" + str(TN))
print(" False Negative =" + str(FN))

Accuracy = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1 = 2*(1/((1/Precision)+(1/Recall)))

print(" True Positive =" + str(TP))
print(" False Positive =" + str(FP))
print(" True Negative =" + str(TN))
print(" False Negative =" + str(FN))
print(" Accuracy = " + str(Accuracy))
print(" Precision = " + str(Precision))
print(" Recall = "+ str(Recall))
print(" F1 Score = " + str(F1))

# print(Vocabulary)