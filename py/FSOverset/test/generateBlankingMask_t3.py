# No offset
from FSDataManager import FSClac, FSError, FSDataManager
from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import generateBlankingMask
import KCore.test as Ktest

localDirIn = './INPUT/'
localDirOut = './OUTPUT/'

globalClac = FSClac()  # by default, FSClac uses MPI_COMM_WORLD, i.e. all processes available
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 3:
    raise ValueError("howToUseCassiopeeHoleCuttingTwoBodies must be run with at least 3 MPI processes.")

globalProcID = globalClac.GetProcID()
meshID = int(globalProcID // (nGlobalProcs / 3))

clac = FSClac()
globalClac.DivideIntoGroups(meshID, clac)

if meshID == 0:
    meshFilename = localDirIn+'background.h5'
    meshKeyOriginal = 'back_orig'  # the original background mesh
    meshKeyActive = 'back_active'  # the active part of the background mesh
elif meshID == 1:
    meshFilename = localDirIn+'naca.h5'
    meshKeyOriginal = 'naca'
    meshKeyActive = 'naca'
else:
    meshFilename = localDirIn+'cylinder_small.h5'
    meshKeyOriginal = 'cyl'
    meshKeyActive = 'cyl'


dictOfOffsets = {}
dictOfOffsets[1] = 0.
dictOfOffsets[2] = 0.

dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOriginal, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh - useful for CODA computations where the active part is extracted
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)

mask = generateBlankingMask(
    clac=clac, fsmesh=fsmeshOrig,
    dictOfOffsets=dictOfOffsets,
    meshID=meshID,
    localDir=localDirOut,
    offsetFromBC='BCWall',  
    check=True
)

for item in mask:
    Ktest.testT(mask[item],item+ globalProcID*len(mask))
