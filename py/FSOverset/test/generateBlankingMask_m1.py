# Usage: kpython -n2 -t4 generateBlankingMask_m1.py
from FSDataManager import FSClac, FSError, FSDataManager
from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import generateBlankingMask
import KCore.test as Ktest

localDirIn = './INPUT/'

offsetDict = {}
offsetDict[1] = 0.3
offsetDict[0] = 0.01 # there is no BCWall in meshID==0 so it will not be taken into account

globalClac = FSClac()  # by default, FSClac uses MPI_COMM_WORLD, i.e. all processes available
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 2:
    raise ValueError('generateBlankingMask_m1.py requires 2 processes at least.')

globalProcID = globalClac.GetProcID()
meshID = int(globalProcID // (nGlobalProcs / 2))

clac = FSClac()
globalClac.DivideIntoGroups(meshID, clac)

if meshID == 0:
    meshFilename = localDirIn+'background.h5'
    meshKeyOrig = 'back_orig'
    meshKeyActive = 'back_active'
else:
    meshFilename = localDirIn+'naca.h5'
    meshKeyOrig = 'naca'
    meshKeyActive = 'naca'

dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOrig, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Automatically generate blankingMaskDict from offsetDict
blankingMaskDict = generateBlankingMask(
    clac=clac, fsmesh=fsmeshOrig,
    offsetDict=offsetDict,
    meshID=meshID,
    offsetFromBC='BCWall', 
    check=False)

# test
if globalProcID == 0:
    for item in blankingMaskDict: Ktest.testT(blankingMaskDict[item],item)

