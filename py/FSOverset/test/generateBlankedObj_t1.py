# Usage: kpython -n2 -t4 howToUseCassiopeeHoleCutting.py
from FSDataManager import FSClac, FSError, FSDataManager

from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset
import Converter.PyTree as C
import KCore.test as Ktest

offsets = [0.3]
localDirIn = './INPUT/'
localDirOut = './OUTPUT/'

globalClac = FSClac()  # by default, FSClac uses MPI_COMM_WORLD, i.e. all processes available
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 2:
    raise ValueError("howToUseCassiopeeHoleCutting must be run with at least 2 MPI processes.")

globalProcID = globalClac.GetProcID()
meshID = int(globalProcID // (nGlobalProcs / 2))

clac = FSClac()
globalClac.DivideIntoGroups(meshID, clac)

if meshID == 0:
    meshFilename = localDirIn+'background.h5'
    meshKeyOrig = 'back_orig'  # the original background mesh
    meshKeyActive = 'back_active'  # the active part of the background mesh
else:
    meshFilename = localDirIn+'naca.h5'
    meshKeyOrig = 'naca'
    meshKeyActive = 'naca'

dictOfOffsets = {}
dictOfOffsets[1] = offsets[0]
dictOfOffsets[0] = 0.01 # there is no BCWall so it won t be taken into account

dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOrig, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh - useful for CODA computations where the active part is extracted
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)

blankedObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, meshIDTarget=0)
t = blankedObj.pyTree
if t is None: t = C.newPyTree(["DUMMY_%d" %meshID])
Ktest.testT(t, meshID)