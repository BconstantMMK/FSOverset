# Usage: kpython -n2 -t4 extractActiveSubMesh_m1.py
from FSDataManager import FSClac, FSError, FSDataManager
from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset, extractActiveSubMesh
from FSPlugins.test import testH5
import Generator.PyTree as G
import Post.PyTree as P
import KCore.test as Ktest

localDirIn = './INPUT/'

# blankingMaskDict: blanking bodies
blankingMaskDict = {}
blankingMaskDict[1] = P.exteriorFaces(G.cart((-0.3,0,-0.2), (2.92,0.02,0.4), (2,2,2)))

# blankingDict: which meshID is blanked by which masks
blankingDict = {}
blankingDict[0] = [1]

globalClac = FSClac()  # by default, FSClac uses MPI_COMM_WORLD, i.e. all processes available
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 2:
    raise ValueError('extractActiveSubMesh_m1 requires 2 processes at least.')

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

# Get active mesh - useful for CODA computations where the active part is extracted
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)

# Initialize FSOverset class for every meshID
blankedObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, blankingDict=blankingDict)

# Update cell nature field with 0 (blanked) and 1 (active)
blankedObj.computeBlanking(blankingMaskDict)

# Remove blanked cells from active mesh
extractActiveSubMesh(dm, meshKeyOrig, meshKeyActive)

# test
testDir = Ktest.getDataFolderName()
testFile = testDir+'/extractActiveSubMesh_m1_%d.h5'%globalProcID
if not testFile: blankedObj.fsmesh.ExportMeshHDF5(Filename=testFile) or FSError.PrintAndExit()
testH5(clac, blankedObj.fsmesh, number=1,
    checkCoordinates=True, coordsName='Coordinates',
    checkConnectivity=True, checkDatasets=True,
    rtol=0., atol=1.e-10,
    reference=testFile)