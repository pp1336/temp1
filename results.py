import DAPICourseworkLibrary as l
import DAPICourseworkSkeleton as s

data = l.ReadFile('Neurones.txt')

print('read data = ')
print(data)

[noVariables, noRoots, noStates, noDataPoints, datain] = data

print('noStates = ')
print(noStates)

root = 0
prior_1 = s.Prior(data, root, noStates)
print('prior_1 = 1')
print(prior_1)
