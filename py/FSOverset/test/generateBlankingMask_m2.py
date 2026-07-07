# Usage: kpython -n3 -t4 generateBlankingMask_m2.py
from FSDataManager import FSError, FSDataManager
from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import generateBlankingMask, getClacInfo
import KCore.test as Ktest

localDirIn = './INPUT/'

meshDict = {
    0: {'meshFilename': localDirIn+'background.h5', 'meshProcessorWeight': 1., 'meshKey':'background'},
    1: {'meshFilename': localDirIn+'naca.h5', 'meshProcessorWeight': 1., 'meshKey':'naca'},
    2: {'meshFilename': localDirIn+'cylinder_small.h5', 'meshProcessorWeight': 1., 'meshKey':'cyl'}
}
offsetDict = {
    1: 0.2,
    2: 0.05
}

# Get clacs
meshID, clac, globalClac, masterClac = getClacInfo(meshDict)
meshFilename = meshDict[meshID]['meshFilename']
meshKey = meshDict[meshID]['meshKey']

# Get orig mesh
dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKey, clac, True)
meshOps = buildMeshOps(meshFilename, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Automatically generate blankingMaskDict from offsetDict
blankingMaskDict = generateBlankingMask(
    clac=clac, fsmesh=fsmeshOrig,
    offsetDict=offsetDict,
    meshID=meshID,
    offsetFromBC='BCWall', 
    check=False)

# test
if globalClac.GetProcID() == 0:
    for item in blankingMaskDict: Ktest.testT(blankingMaskDict[item],item)