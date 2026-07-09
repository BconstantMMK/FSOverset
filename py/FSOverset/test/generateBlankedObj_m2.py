# Usage: kpython -n3 -t4 generateBlankedObj_m2.py
from FSDataManager import FSError, FSDataManager
from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset, getClacInfo, getMeshKeys
import Converter.PyTree as C
import KCore.test as Ktest

localDirIn = './INPUT/'

meshDict = {
    0: {'meshFilename': localDirIn+'background.h5', 'meshProcessorWeight': 1., 'meshKey':'background'},
    1: {'meshFilename': localDirIn+'naca.h5', 'meshProcessorWeight': 1., 'meshKey':'naca'},
    2: {'meshFilename': localDirIn+'cylinder_large.h5', 'meshProcessorWeight': 1., 'meshKey':'cyl'}
}
blankingDict = {
    0: [1,2],
    2: [1]
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

# test
t = blankedObj.pyTree        
if t is None: t = C.newPyTree(['DUMMY_%d'%meshID])
Ktest.testT(t,globalClac.GetProcID())