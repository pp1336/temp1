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
    # normalise for marginal probabilities
    As = jP_.sum(axis=1)
    Bs = jP_.sum(axis=0)
    # compute mutual information
    for i in xrange(jP_.shape[0]):
        for j in xrange(jP_.shape[1]):
            ab = jP[i][j]
            # ignore zero value
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
        # form upper triangular matrix
        for j in xrange(i + 1, noVariables):
            # form dependency matrix
            jP = JPT(theData, i, j, noStates)
            # calculate mutual infor
            mi = MutualInformation(jP)
            MIMatrix[i][j] = mi
# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    noVariables = depMatrix.shape[0]
    # get depency from the upper triangular part
    for i in xrange(noVariables):
        for j in xrange(i + 1, noVariables):
            # add edge
            depList.append([depMatrix[i][j], i, j])
    # sort according to mutual information
    depList.sort(key = lambda x: x[0], reverse = True)
    depList2 = array(depList)
# end of coursework 2 task 3
    return array(depList2)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
    # stores the graph built so far
    graph = {}
    # build graph and collect edges
    for edge in depList:
        # max number of edges <= no. of vars - 1
        if len(spanningTree) == noVariables - 1:
            break
        n1, n2 = edge[1], edge[2]
        # use BFS to detect if there is existing path
        # between n1 and n2
        if not BFS(graph, n1, n2, len(spanningTree) + 1):
            # update graph and add edge
            addNode(graph, n1, n2)
            addNode(graph, n2, n1)
            spanningTree.append(edge) 
    return array(spanningTree)

# helper functon to add an edge to graph between n1 and n2
def addNode(graph, n1, n2):
    if n1 in graph:
        graph[n1].append(n2)
    else:
        graph[n1] = [n2]

# helper function using BFS to detect whether n2 can be
# reached from n1 by following the edges in graph
def BFS(graph, n1, n2, size):
    # if n1 or n2 not in grpah, then cannot reach
    if not (n1 in graph and n2 in graph):
        return False
    # set of nodes visited so far
    visited = set()
    # queue for bfs
    q = [n1]
    # loop until queue is empty
    while len(q) > 0:
        cur = q.pop(0)
        # avoid double visit
        if cur in visited:
            continue
        visited.add(cur)
        # found path
        if cur == n2:
            return True
        # max number of distinct visit is <= no. nodes in graph
        if len(visited) == size:
            return False
        # get all children nodes
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
