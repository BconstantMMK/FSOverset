# Usage: kpython -n3 -t4 howToUseCassiopeeHoleCuttingTwoBodiesMeshIntersection.py
from FSDataManager import FSClac, FSError, FSDataManager

from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset, generateBlankingMask, extractActiveSubMesh, getClacInfo, getMeshKeys

localDirIn = './INPUT/'
localDirOut = './OUTPUT/TEST3/'

meshDict = {
    0: {'meshFilename': localDirIn+'background.h5', 'meshProcessorWeight': 1., 'meshKey':'background'},
    1: {'meshFilename': localDirIn+'naca.h5', 'meshProcessorWeight': 1., 'meshKey':'naca'},
    2: {'meshFilename': localDirIn+'cylinder_large.h5', 'meshProcessorWeight': 1., 'meshKey':'cyl'},
}
offsetDict = {
    1: 0.3,
    2: 0.5
}
blankingDict = {
    0: [1,2],
    1: [2],
    2: [1]
}

# #===================
# # blanking data - user defined
# offsetDict={}
# offsetDict[1]=0.3
# offsetDict[2]=0.5

# blankingDict={}
# blankingDict[0]=[1,2]
# blankingDict[1]=[2]
# blankingDict[2]= [1]

# Get clacs
meshID, clac, globalClac, masterClac = getClacInfo(meshDict)
meshFilename = meshDict[meshID]['meshFilename']
meshKeyActive, meshKeyOrig = getMeshKeys(meshID, meshDict, blankingDict)

# Get orig mesh
dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOrig, clac, True)
meshOps = buildMeshOps(meshFilename, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)

# FSOverset
blankingMaskDict = generateBlankingMask(
    clac=clac, fsmesh=fsmeshOrig,
    offsetDict=offsetDict,
    meshID=meshID,
    localDir=localDirOut,
    offsetFromBC='BCWall',  
    check=False)

blankingObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, blankingDict=blankingDict)

# 1-update cell nature field with 0 (blanked) and 1 (active)
blankingObj.computeBlanking(blankingMaskDict=blankingMaskDict)

# 2-remove blanked cells
extractActiveSubMesh(dm, meshKeyOrig, meshKeyActive) 

# 3-save active mesh
fsmeshActive.ExportMeshHDF5(Filename=localDirOut+'blanked_mesh_%d.h5'%meshID) or FSError.PrintAndExit()