'''Creates a dataset of all angles, that is readable by the MPI implementation'''

import numpy as np
import random

data = np.genfromtxt("../PythonCode/TestFiles/allPairsCsvs3/!AllCombined4Angles.csv", dtype=str)

protein_list = []
for fam in np.unique(data[:,4]):
    famData = data[data[:,4]==fam]
    for pair in np.unique(famData[:,5]):
        protein_list.append(famData[famData[:,5]==pair])

i = 0
size = []
for subset in protein_list:
    subset = subset[:,(8,10,9,11,-1)].astype(float)
    subset = subset[~np.any(subset==-1000, axis=1)]
    size.append(subset.shape[0])
    if (i==0):
        outMat = np.array(subset);
    else:
        outMat = np.vstack((outMat, subset))
    i += 1

np.savetxt("../Mpi_model/data/FullHomstradSample.dat", outMat, delimiter=",")
np.savetxt("../Mpi_model/data/FullHomstradArray.dat", size, delimiter=",", fmt="%i")
