# Usage: kpython -n2 -t4 computeBlanking_m1.py
from FSDataManager import FSError, FSDataManager
from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset, getClacInfo, getMeshKeys
from FSPlugins.test import testH5
import Generator.PyTree as G
import Post.PyTree as P
import KCore.test as Ktest

localDirIn = './INPUT/'

meshDict = {
    0: {'meshFilename': localDirIn+'background.h5', 'meshProcessorWeight': 1., 'meshKey':'background'},
    1: {'meshFilename': localDirIn+'naca.h5', 'meshProcessorWeight': 1., 'meshKey':'naca'}
}
blankingMaskDict = {
    0: P.exteriorFaces(G.cart((-0.3,0,-0.2), (2.92,0.02,0.4), (2,2,2)))
}
blankingDict = {
    0: [0]
}

# Get clacs
meshID, clac, globalClac, masterClac = getClacInfo(meshDict)
meshFilename = meshDict[meshID]['meshFilename']
meshKeyActive, meshKeyOrig = getMeshKeys(meshID, meshDict, blankingDict)

# Get orig mesh
dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOrig, clac, True)
meshOps = buildMeshOps(meshFilename, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Initialize FSOverset class for every meshID
blankedObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, blankingDict=blankingDict)

# Update cell nature field with 0 (blanked) and 1 (active)
blankedObj.computeBlanking(blankingMaskDict)

# test
testDir = Ktest.getDataFolderName()
testFile = testDir+'/computeBlanking_m1_%d.h5'%globalClac.GetProcID()
if not testFile: blankedObj.fsmesh.ExportMeshHDF5(Filename=testFile) or FSError.PrintAndExit()
testH5(clac, blankedObj.fsmesh, number=1,
    checkCoordinates=True, coordsName='Coordinates',
    checkConnectivity=True, checkDatasets=True,
    rtol=0., atol=1.e-10,
    reference=testFile)