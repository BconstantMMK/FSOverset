# Usage: kpython -n3 -t4 howToUseCassiopeeHoleCuttingTwoBodiesMeshIntersection.py
from FSDataManager import FSClac, FSError, FSDataManager

from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset, generateBlankingMask, extractActiveSubMesh

"""
In this configuration the overset border of the child mesh of the cylinder
intersects the wall of the naca child mesh.
This is not acceptable because this would generate orphan cells (no donor found in CODA).
In this case the idea is to 'cut' the mesh that overlaps the wall of the other
mesh (in this case we cut the cylinder mesh), simply applying the same blanking
procedure that we usually (in the single body case) apply to the background mesh.
"""

localDirIn = './INPUT/'
localDirOut = './OUTPUT/TEST3/'

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
    meshKeyOriginal = 'back_orig'  # the original background mesh
    meshKeyActive = 'back_active'  # the active part of the background mesh
elif meshID == 1:
    meshFilename = localDirIn+'naca.h5'
    meshKeyOriginal = 'naca_orig'  # the original naca mesh
    meshKeyActive = 'naca_active'
elif meshID == 2:
    meshFilename = localDirIn+'cylinder_large.h5'
    meshKeyOriginal = 'cyl_orig' # now the cylinder is also blanked
    meshKeyActive = 'cyl_active'

#===================
# blanking data - user defined
dictOfOffsets={}
dictOfOffsets[1]=0.3
dictOfOffsets[2]=0.5

dictOfBlanking={}
dictOfBlanking[0]=[1,2]
dictOfBlanking[1]=[2]
dictOfBlanking[2]= [1]

# MANDATORY to set to 'none' for non-blanked meshes for extractActiveSubMesh to work properly
# need to be before the initialization of the fsMeshActive !!!
if meshID not in dictOfBlanking:
    meshKeyOriginal = 'none'
    meshKeyActive = 'none'

# ====================================
dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOriginal, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh - useful for CODA computations where the active part is extracted
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)

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
