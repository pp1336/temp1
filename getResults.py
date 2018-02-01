import DAPICourseworkLibrary as l
import DAPICourseworkSkeleton as s
from numpy import *


Variables, noRoots, noStates, noDataPoints, datain = l.ReadFile("Neurones.txt")
theData = array(datain)
prior = s.Prior(theData, 1, noStates)
print('prior = ')
print(prior)
print('sum = ')
print(sum(prior))
