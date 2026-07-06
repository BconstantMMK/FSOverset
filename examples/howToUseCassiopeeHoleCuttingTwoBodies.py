# Usage: kpython -n3 -t4 howToUseCassiopeeHoleCuttingTwoBodies.py
from FSDataManager import FSClac, FSError, FSDataManager

from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset, generateBlankingMask, extractActiveSubMesh

"""
The following three cases have the SAME treatment, both in the blanking process and in CODA.
In case the blanked area of a child mesh intersects the WALL of the other child mesh,
refer to example howToUseCassiopeeHoleCuttingTwoBodiesDoubleBlanking.py.
"""

localDirIn = './INPUT/'
localDirOut = './OUTPUT/TEST2/'

#===================
# blanking data - user defined
offsetDict={}
offsetDict[1]=0.3
offsetDict[2]=0.1

blankingDict = {}
blankingDict[0]= [1,2]
blankingDict[2]= [1]

#=============================================================================================
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
    meshKeyOriginal = 'naca_orig'
    meshKeyActive = 'naca_active'
else:
    meshFilename = localDirIn+'cylinder_small.h5'
    meshKeyOriginal = 'cyl_orig'
    meshKeyActive = 'cyl_active'

# MANDATORY to set to 'none' for non-blanked meshes for extractActiveSubMesh to work properly
# need to be before the initialization of the fsMeshActive !!!
if meshID not in blankingDict:
    meshKeyOriginal = 'none'
    meshKeyActive = 'none'

## ====================================
## Create Data Manager & set Original/Active
## ====================================    
dm = FSDataManager(globalClac)

fsmeshOrig = dm.GetMesh(meshKeyOriginal, clac, True)
# Get active mesh - useful for CODA computations where the active part is extracted
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)

meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

#blanking objs creation
blankingMaskDict = generateBlankingMask(
    clac=clac, fsmesh=fsmeshOrig,
    offsetDict=offsetDict,
    meshID=meshID,
    localDir=localDirOut,
    offsetFromBC='BCWall',  
    check=False)
blankedObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, blankingDict=blankingDict)

# FROM NOW ON, THIS CAN BE WITHIN A TIME STEP LOOP
# 1-update cell nature field with 0 (blanked) and 1 (active)
blankedObj.computeBlanking(blankingMaskDict=blankingMaskDict)

# 2-remove blanked cells
extractActiveSubMesh(dm, meshKeyOriginal, meshKeyActive) 
fsmeshActive.ExportMeshHDF5(Filename="blanked_mesh_%d.h5"%meshID) or FSError.PrintAndExit()
