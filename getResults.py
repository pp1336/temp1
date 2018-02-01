import DAPICourseworkLibrary as l
import DAPICourseworkSkeleton as s
import numpy
from numpy import *


Variables, noRoots, noStates, noDataPoints, datain = l.ReadFile("Neurones.txt")
theData = array(datain)
prior = s.Prior(theData, 1, noStates)
print('prior = ')
print(prior)
print('sum = ')
print(sum(prior))
print('==========================================')

varC = 0
varP = 1
CPT = s.CPT(theData, varC, varP, noStates)
print('size should be ' + str(noStates[varC]) + ' by ' + str(noStates[varP]))
print('got ' + str(len(CPT)) + ' by ' + str(len(CPT[0])))
print('cpt = ')
print(CPT)
print('sum = ')
CPT = numpy.array(CPT)
print(CPT)
print(CPT.sum(axis=0))
print('==========================================')

varRow = 1
varCol = 2
CPT = s.JPT(theData, varRow, varCol, noStates)
print('size should be ' + str(noStates[varRow]) + ' by ' + str(noStates[varCol]))
print('got ' + str(len(CPT)) + ' by ' + str(len(CPT[0])))
print('cpt = ')
print(CPT)
print('sum = ')
CPT = numpy.array(CPT)
print(CPT)
print(sum(CPT))
print('==========================================')

varC = 2
varP = 1
JPT = s.JPT(theData, varC, varP, noStates)
CPT = s.CPT(theData, varC, varP, noStates)
CPT_ = s.JPT2CPT(JPT)
print(array(CPT))
print(array(CPT_))
print('equal result')
dif = (array(CPT) - array(CPT_)) ** 2
print(dif)
print(sum(dif))
print('==========================================')

naiveBayes = [array(s.Prior(theData, 0, noStates))]
for i in xrange(1, 6):
    naiveBayes.append(array(s.CPT(theData, i, 0, noStates)))
theQuery =  [4,0,0,0,5] #[6, 5, 2, 5, 5]
rootPdf = s.Query(theQuery, naiveBayes)
print('rootPdf = ')
print(rootPdf)
print(theQuery)
print(naiveBayes[0])
for i in xrange(1, 6):
    print(naiveBayes[i][theQuery[i - 1]])
