#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from DAPICourseworkLibrary import *
from numpy import *
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
# Coursework 1 task 1 should be inserted here
    for dataPoint in theData:
        prior[dataPoint[root]] += 1
    total = float(len(theData))
    prior = [freq / total for freq in prior]
# end of Coursework 1 task 1
    return prior
# Function to compute a CPT with parent node varP and child node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
# Coursework 1 task 2 should be inserte4d here
    varPcount = zeros(noStates[varP])
    for dataPoint in theData:
        cPT[dataPoint[varC], dataPoint[varP]] += 1
        varPcount[dataPoint[varP]] += 1
    varPcount = [float(x) for x in varPcount]
    cPT = [[col / total for col, total in zip(row, varPcount)] for row in cPT]
# end of coursework 1 task 2
    return cPT
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here 
    for dataPoint in theData:
        jPT[dataPoint[varRow], dataPoint[varCol]] += 1
    total = float(len(theData))
    jPT = [[col / total for col in row] for row in jPT]
# end of coursework 1 task 3
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here
    aJPT_ = array(aJPT)
    Bs = aJPT_.sum(axis=0)
    aJPT = [[col / bs for col, bs in zip(row, Bs)] for row in aJPT]
# coursework 1 taks 4 ends here
    return aJPT

#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
    print(rootPdf.shape)
    print(naiveBayes)
# Coursework 1 task 5 should be inserted here
    for i in xrange(rootPdf.shape[0]):
        p = naiveBayes[0][i]
        for j in xrange(1, len(naiveBayes)):
            p *= naiveBayes[j][theQuery[j - 1]][i]
        rootPdf[i] = p
    print(rootPdf)
    total = float(sum(rootPdf))
    rootPdf = [l / total for l in rootPdf]
# end of coursework 1 task 5
    return rootPdf
#
# End of Coursework 1
#

#
# main program part for Coursework 1
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
theData = array(datain)
AppendString("results.txt","Coursework One Results by dfg")
AppendString("results.txt","") #blank line

AppendString('results.txt','The prior probability of node 0')
prior = Prior(theData, 0, noStates)
AppendList('results.txt', array(prior))

AppendString('results.txt', 'The conditional probability matrix P(2|0) calculated from the data')
cpt = CPT(theData, 2, 0, noStates)
AppendArray('results.txt', array(cpt))

AppendString('results.txt', 'The joint probability matrix P(2&0) calculated from the data')
jpt = JPT(theData, 2, 0, noStates)
AppendArray('results.txt', array(jpt))

AppendString('results.txt', 'The conditional probability matrix P(2|0) calculated from the joint probability matrix P(2&0)')
cpt_ = JPT2CPT(jpt)
AppendArray('results.txt', array(cpt))

AppendString('results.txt', 'The results of queries [4,0,0,0,5] on the naive network')
naiveBayes = [array(prior)]
for i in xrange(1, 6):
    naiveBayes.append(array(CPT(theData, i, 0, noStates)))
theQuery =  [4,0,0,0,5]
pdf1 = Query(theQuery, naiveBayes)
theQuery =  [6, 5, 2, 5, 5]
pdf2 = Query(theQuery, naiveBayes)
AppendList('results.txt', array(pdf1))
AppendString('results.txt', 'The results of queries [6, 5, 2, 5, 5] on the naive network')
AppendList('results.txt', array(pdf2))
#
#
