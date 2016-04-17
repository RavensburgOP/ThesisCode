'''Compares pairs created using fasttree in the same way as the report
for the Blok 4 project'''

import os, csv
import numpy as np

def Parser(csv_file):
    PairList = []
    with open(csv_file, 'rb') as read_file:
        reader = csv.reader(read_file)
        for row in reader:
            PairList.append(row)
    return PairList

def makeArray(PairList, famName):
    print famName
    TotalCompos = 0
    for i in range(len(PairList)/11):
        offset = i * 11
        # Time to last common ancestor
        TTLCA = PairList[0+offset][0].split(" ")[2]

        # Phi angles for each position
        Phi0 = np.array(PairList[3+offset], dtype="float")
        Phi1 = np.array(PairList[5+offset], dtype="float")
        # Find if either is missing an angle
        NoPhi = np.logical_or(Phi0 == -1000, Phi1 == -1000)

        # Delta angles
        DeltaPhi = Phi0 - Phi1
        DeltaPhi = (DeltaPhi+np.pi)%(2*np.pi)-np.pi
        DeltaPhi[NoPhi] = -1000

        # Psi angles for all positions
        Psi0 = np.array(PairList[4+offset], dtype="str")
        Psi1 = np.array(PairList[6+offset], dtype="str")

        # Change strings to floats to make comparison possible
        Psi0[Psi0 == "None"] = -1000
        Psi1[Psi1 == "None"] = -1000
        Psi0 = Psi0.astype("float")
        Psi1 = Psi1.astype("float")

        # Find all missing angles
        NoPsi = np.logical_or(Psi0 == -1000, Psi1 == -1000)

        # Calculate delta angles
        DeltaPsi = Psi0 - Psi1
        DeltaPsi = (DeltaPsi+np.pi)%(2*np.pi)-np.pi
        DeltaPsi[NoPsi] = -1000

        AAarray = np.empty([2, len(PairList[9][0])], dtype="str")
        SSarray = np.empty([2, len(PairList[9][0])], dtype="str")

        missAngles = np.logical_and(NoPsi, NoPhi)

        # Get the alignment correct
        for k in range(offset+1, offset+3):
            count = 0
            ProtNumber = k-1-offset
            # Index 8 is the alignment info. 6 is the secondary structure.
            for j, ali in enumerate(PairList[k+8][0]):
                if ali == '#':
                    AAarray[ProtNumber][j] = PairList[k][0][count]
                    SSarray[ProtNumber][j] = list(PairList[k+6][0])[count]
                    count += 1
                else:
                    AAarray[ProtNumber][j] = "-"
                    SSarray[ProtNumber][j] = "-"
        # find common positions ("and" is used because it's not possible to
        # determine whether and insertion or deletion event has happend)
        ComPos = np.logical_and(AAarray[0] != '-', AAarray[1] != '-')

        # Secondary structure
        # PairList[]

        # Add to sum of indels in the family
        TotalCompos += np.sum(ComPos)

        # Count number of different neighbours
        NotSameAA = (AAarray[0] != AAarray[1])[ComPos]
        num_neighbor = 5

        num_neighbor_list = []
        num_changes_list = []
        for index in range(len(NotSameAA)):
            # Boundaries
            left = max(0,index-num_neighbor)
            right = min(num_neighbor+index,len(NotSameAA))
            # How many changes
            changes = np.sum(NotSameAA[left:right])
            num_changes_list.append(changes)
            # Out of how many
            AA_neighbors = right-left
            num_neighbor_list.append(AA_neighbors)

        glob_changes = [np.sum(NotSameAA)]*np.sum(ComPos)
        glob_len = [len(NotSameAA)]*np.sum(ComPos)

        # Output array. First all components are added, then stacked into a
        # huge array
        tempArray = np.array([[famName]*np.sum(ComPos), [i]*np.sum(ComPos), DeltaPhi[ComPos], DeltaPsi[ComPos], Phi0[ComPos], Phi1[ComPos], Psi0[ComPos], Psi1[ComPos], num_changes_list, num_neighbor_list, glob_changes, glob_len, [TTLCA]*np.sum(ComPos)]).T
        tempArray = np.hstack((AAarray.T[ComPos],tempArray))
        tempArray = np.hstack((SSarray.T[ComPos],tempArray))

        # If it's the first iteration, an array is created. All subsequent are
        # stacked with this array
        if i == 0:
            OutputArray = tempArray
        else:
            OutputArray = np.vstack((OutputArray, tempArray))
        if OutputArray.shape[1] != 17:
            break

    return (OutputArray, TotalCompos)

pathToDb = 'PairedStructures/'

csvFiles = [i for i in os.listdir(pathToDb) if i.endswith(".csv")]
first = True
TNumComPos = 0

for csvFile in csvFiles:
    famName = os.path.splitext(os.path.basename(csvFile))[0]
    testPair = Parser(pathToDb+csvFile)
    output, NumComPos = makeArray(testPair, famName)
    TNumComPos += NumComPos
    if first:
        Combined = output
        first = False
    else:
        Combined = np.vstack((Combined, output))
    np.savetxt("allPairsCsvs/"+csvFile, output, fmt="%s")

print TNumComPos
np.savetxt("allPairsCsvs/!AllCombined4Angles.csv", Combined, fmt="%s")
