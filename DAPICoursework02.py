#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from DAPICourseworkLibrary import *
from numpy import *
import math
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
# Coursework 1 task 1 should be inserted here
    for dataPoint in theData:
        # greq of data
        prior[dataPoint[root]] += 1
    # normalise
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
        # co-occurance
        cPT[dataPoint[varC], dataPoint[varP]] += 1
        # freq of parent
        varPcount[dataPoint[varP]] += 1
    # normalise
    varPcount = [float(x) for x in varPcount]
    cPT = [[col / total for col, total in zip(row, varPcount)] for row in cPT]
# end of coursework 1 task 2
    return cPT

# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here 
    for dataPoint in theData:
        # freq of co-occurance
        jPT[dataPoint[varRow], dataPoint[varCol]] += 1
    # normalise
    total = float(len(theData))
    jPT = [[col / total for col in row] for row in jPT]
# end of coursework 1 task 3
    return jPT

# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here
    aJPT_ = array(aJPT)
    # sum each column
    Bs = aJPT_.sum(axis=0)
    # normalise
    aJPT = [[col / bs for col, bs in zip(row, Bs)] for row in aJPT]
# coursework 1 taks 4 ends here
    return aJPT

# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
# Coursework 1 task 5 should be inserted here
    for i in xrange(rootPdf.shape[0]):
        # get the prior for i
        p = naiveBayes[0][i]
        # multiply by p(j|i) for each j
        for j in xrange(1, len(naiveBayes)):
            p *= naiveBayes[j][theQuery[j - 1]][i]
        rootPdf[i] = p
    # normalise
    total = float(sum(rootPdf))
    rootPdf = [l / total for l in rootPdf]
# end of coursework 1 task 5
    return rootPdf
#
# End of Coursework 1
#

#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
# Coursework 2 task 1 should be inserted here
    jP_ = array(jP)
    As = jP_.sum(axis=1)
    Bs = jP_.sum(axis=0)
    for i in xrange(jP_.shape[0]):
        for j in xrange(jP_.shape[1]):
            ab = jP[i][j]
            if ab != 0:
                mi += ab * math.log(ab / (As[i] * Bs[j]), 2)
# end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    for i in xrange(noVariables):
        for j in xrange(i + 1):
            jP = JPT(theData, i, j, noStates)
            mi = MutualInformation(jP)
            MIMatrix[i][j] = mi
            if i != j:
                MIMatrix[j][i] = mi
# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    noVariables = depMatrix.shape[0]
    for i in xrange(noVariables):
        for j in xrange(i):
            depList.append([depMatrix[i][j], i, j])
    depList.sort(key = lambda x: x[0], reverse = True)
    depList2 = array(depList)
# end of coursework 2 task 3
    return array(depList2)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
    graph = {}
    for edge in depList:
        if len(spanningTree) == noVariables - 1:
            break
        n1, n2 = edge[1], edge[2]
        if not BFS(graph, n1, n2, len(spanningTree) + 1):
            addNode(graph, n1, n2)
            addNode(graph, n2, n1)
            spanningTree.append(edge) 
    return array(spanningTree)

def addNode(graph, n1, n2):
    if n1 in graph:
        graph[n1].append(n2)
    else:
        graph[n1] = [n2]

def BFS(graph, n1, n2, size):
    if not (n1 in graph and n2 in graph):
        return False
    visited = set()
    q = [n1]
    while len(q) > 0:
        cur = q.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == n2:
            return True
        if len(visited) == size:
            return False
        for n in graph[cur]:
            q.append(n)
    return False
#
# End of coursework 2
#

#
# main program part for Coursework 2
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
AppendString("results.txt", "Coursework Two Results by dfg")
AppendString("results.txt", "") #blank line

jP = JPT(theData, 1, 2, noStates)

dep_matrix = DependencyMatrix(theData, noVariables, noStates)
AppendString("results.txt", "The dependency matrix for the HepatitisC data set")
AppendArray("results.txt", dep_matrix)

dep_list = DependencyList(dep_matrix)
AppendString("results.txt", "The dependency list for the HepatitisC data set")
AppendArray("results.txt", dep_list)

spanning_tree = SpanningTreeAlgorithm(dep_list, noVariables)
AppendString("results.txt", "The spanning tree found for the HepatitisC data set")
AppendArray("results.txt", spanning_tree)
#
#
