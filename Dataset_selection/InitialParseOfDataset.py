import imp
from Bio.Phylo.Applications import _Fasttree
import Bio.Phylo as Phylo
from Bio import SeqIO
import numpy as np
import csv
import os
from glob import glob

# A copy of the homstrad database must be aquired and the path to the folder specified here:
pathToDB = "<PATH TO HOMSTRAD>"

# Path to FastTree executable
fasttree_exe = "<PATH TO FastTree>"

pathToMain = "HOMSTRAD_parser/main.py"
Parser = imp.load_source("Parser", pathToMain)

# Find the path to each family
pdb = [y for x in os.walk(pathToDB) for y in glob(os.path.join(x[0], '*.pdb'))]
ali = [y for x in os.walk(pathToDB) for y in glob(os.path.join(x[0], '*.ali'))]
families = zip(pdb, ali)

# load parser class
Parser = imp.load_source("Parser", pathToMain)

num_pairs = 0

for family in families:
    famName = os.path.splitext(os.path.basename(family[1]))[0]
    print famName
    # Tree part
    # Convert file so fasttree can use it
    sequences = SeqIO.convert(family[1], "pir", "temp.fasta", "fasta")

    cmd = _Fasttree.FastTreeCommandline(fasttree_exe, input="temp.fasta", out='ExampleTree.tree')

    cmd()

    # Load into Python
    tree = Phylo.read('ExampleTree.tree', "newick")

    cherryX = []
    cherryY = []

    # Find leaf pairs
    for i in tree.find_clades():
        # Is node before leaves
        if i.is_preterminal():
            # Can't use len for iter. Needs list.
            sub_tree = list(i.find_clades())
            if len(sub_tree) == 3:
                j1, j2, j3 = sub_tree
                if j2.branch_length > j3.branch_length:
                    cherryY.append((j2, j3, j1.distance(j2, j3)))
                else:
                    cherryY.append((j3, j2, j1.distance(j2, j3)))
            else:
                print "Bad cluster: ",famName
    if len(cherryY) == 0:
        print "Not enough pairs"
        continue

    # Output part
    Globin = Parser.AlignStruc(family[0], family[1])
    Globin.GetContainedStructures()

    Globin.PirPdbMatch()
    if not Globin.isWorking():
        continue

    print cherryY
    toCsvList = []
    for pair in cherryY:
        num_pairs += 1
        pairList = [np.NAN]*15
        # distance between pair
        pairList[2] = str(pair[2])
        for i, clade in enumerate(pair[:2]):
            key = str(clade)
            pairList[0+i] = Globin[key].Name
            pairList[3+i] = str(Globin[key].rFactor)
            pairList[5+i] = Globin[key].getPdbSeq()
            phiPsiAA = np.array(Globin[key].getPhiPsiAA())
            pairList[7+(2*i)] = ",".join(phiPsiAA[:,0].flatten().astype(str))
            pairList[8+(2*i)] = ",".join(phiPsiAA[:,1].flatten().astype(str))

            # Placeholder secondary structure
            pairList[11+i] = Globin[key].SS

            # Replace this with regex
            homStradAlign = ''
            for char in Globin[key].seq:
                if not char == '-':
                    homStradAlign += '#'
                else:
                    homStradAlign += char
            pairList[13+i] = homStradAlign
        pairList[4] = " ".join(pairList[:5])
        toCsvList.append(pairList[4:])

    # Save dataset
    filename = "PairedStructures/"+famName+".csv"
    with open(filename, "wb") as csvfile:
        writer = csv.writer(csvfile, delimiter="\n")
        writer.writerows(toCsvList)
print num_pairs
