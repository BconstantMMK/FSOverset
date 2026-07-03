# Usage: kpython -n3 -t4 generateBlankingMask_m3.py
from FSDataManager import FSClac, FSError, FSDataManager
from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import generateBlankingMask
import KCore.test as Ktest

localDirIn = './INPUT/'

dictOfOffsets = {}
dictOfOffsets[1] = 0.
dictOfOffsets[2] = 0.

globalClac = FSClac()  # by default, FSClac uses MPI_COMM_WORLD, i.e. all processes available
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 3:
    raise ValueError('generateBlankingMask_m3.py requires 3 processes at least.')

globalProcID = globalClac.GetProcID()
meshID = int(globalProcID // (nGlobalProcs / 3))

clac = FSClac()
globalClac.DivideIntoGroups(meshID, clac)

if meshID == 0:
    meshFilename = localDirIn+'background.h5'
    meshKeyOrig = 'back_orig'
    meshKeyActive = 'back_active'
elif meshID == 1:
    meshFilename = localDirIn+'naca.h5'
    meshKeyOrig = 'naca'
    meshKeyActive = 'naca'
else:
    meshFilename = localDirIn+'cylinder_small.h5'
    meshKeyOrig = 'cyl'
    meshKeyActive = 'cyl'

dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOrig, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Automatically generate dictOfMasks from dictOfOffsets
dictOfMasks = generateBlankingMask(
    clac=clac, fsmesh=fsmeshOrig,
    dictOfOffsets=dictOfOffsets,
    meshID=meshID,
    offsetFromBC='BCWall', 
    check=False)

# test
if globalProcID == 0:
    for item in dictOfMasks: Ktest.testT(dictOfMasks[item],item)
