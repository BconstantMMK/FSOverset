# Usage: kpython -n3 -t4 howToUseCassiopeeHoleCuttingTwoBodiesMeshIntersection.py
from FSDataManager import FSClac, FSError, FSDataManager
from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset, generateBlankingMask
import Generator.PyTree as G
import Post.PyTree as P
import Converter.Mpi as Cmpi
import Converter.PyTree as C
import KCore.test as Ktest

offsets = [0.3, 0.1]

localDirIn = './INPUT/'

globalClac = FSClac()  # by default, FSClac uses MPI_COMM_WORLD, i.e. all processes available
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 3:
    raise ValueError('howToUseCassiopeeHoleCuttingTwoBodiesMeshIntersection must be run with at least 3 MPI processes.')

globalProcID = globalClac.GetProcID()
meshID = int(globalProcID // (nGlobalProcs / 3))

clac = FSClac()
globalClac.DivideIntoGroups(meshID, clac)

discParaDict = {}
wallBoundaryMarkers = []
if meshID == 0:
    meshFilename = localDirIn+'background.h5'
    meshKeyOrig = 'back_orig'  # the original background mesh
    meshKeyActive = 'back_active'  # the active part of the background mesh
elif meshID == 1:
    meshFilename = localDirIn+'naca.h5'
    meshKeyOrig = 'naca'
    meshKeyActive = 'naca'
elif meshID == 2:
    meshFilename = localDirIn+'cylinder_large.h5'
    meshKeyOrig = 'cyl_orig' # now the cylinder is also blanked
    meshKeyActive = 'cyl_active'

dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOrig, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh - useful for CODA computations where the active part is extracted
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)


dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOrig, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh - useful for CODA computations where the active part is extracted
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)

mask={}
meshIDTarget = 0
mask[meshIDTarget] = P.exteriorFaces(G.cart((-0.3,0,-0.2), (2.92,0.02,0.4), (2,2,2)))
blankedObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, meshIDTarget=meshIDTarget)
# 1-update cell nature field with 0 (blanked) and 1 (active)
blankedObj.computeBlanking(mask)
#extractActiveSubMesh(dm, meshKeyOrig, meshKeyActive)
testDir = Ktest.getDataFolderName()
testFile = testDir+'/computeBlanking_t1_%d_%d_%d.h5'%(meshID, meshIDTarget, Cmpi.rank)
if not testFile: blankedObj.fsmesh.ExportMeshHDF5(Filename=testFile) or FSError.PrintAndExit()
from FSPlugins.test import testH5
testH5(clac, blankedObj.fsmesh, number=1,
    checkCoordinates=True, coordsName="Coordinates",
    checkConnectivity=True, checkDatasets=True,
    rtol=0., atol=1.e-10,
    reference=testFile)