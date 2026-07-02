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
    meshKeyOriginal = 'cyl_orig'
    meshKeyActive = 'cyl_active'

#===================
# blanking data - user defined
dictOfOffsets={}
dictOfOffsets[1]=0.3
dictOfOffsets[2]=0.1

dictOfBlanking = {}
dictOfBlanking[0]= [1,2]
dictOfBlanking[2]= [1]

# MANDATORY to set to 'none' for non-blanked meshes for extractActiveSubMesh to work properly
# need to be before the initialization of the fsMeshActive !!!
if meshID not in dictOfBlanking:
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
dictOfMasks = generateBlankingMask(
    clac=clac, fsmesh=fsmeshOrig,
    dictOfOffsets=dictOfOffsets,
    meshID=meshID,
    localDir=localDirOut,
    offsetFromBC='BCWall',  
    check=False)
blankedObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, dictOfBlanking=dictOfBlanking)

# FROM NOW ON, THIS CAN BE WITHIN A TIME STEP LOOP
# 1-update cell nature field with 0 (blanked) and 1 (active)
blankedObj.computeBlanking(dictOfMasks=dictOfMasks)

# 2-remove blanked cells
extractActiveSubMesh(dm, meshKeyOriginal, meshKeyActive) 
fsmeshActive.ExportMeshHDF5(Filename="blanked_mesh_%d.h5"%meshID) or FSError.PrintAndExit()
