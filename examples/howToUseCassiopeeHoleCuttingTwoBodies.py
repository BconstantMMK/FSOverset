# Usage: kpython -n3 -t4 howToUseCassiopeeHoleCuttingTwoBodies.py
from FSDataManager import FSClac, FSError, FSDataManager

from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset, generateBlankingMask, extractPyTree, extractActiveSubMesh

"""
The following three cases have the SAME treatment, both in the blanking process and in CODA.
In case the blanked area of a child mesh intersects the WALL of the other child mesh,
refer to example howToUseCassiopeeHoleCuttingTwoBodiesDoubleBlanking.py.
"""

# offsets = [0.03,0.05] # case in which there is no intersection, neither of the blanked areas of the two child meshes nor of a child mesh overset border with blanked area of the other child mesh
offsets = [0.2,0.05] # case in which the blanked areas of the two child meshes do not intersect, but the cylinder child mesh overset border intersects the blanked area of the naca child mesh
# offsets = [0.3,0.5]  # case in which the blanked areas of naca and cylinder child mesh intersect

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
    meshKeyOriginal = 'cyl'
    meshKeyActive = 'cyl'

dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOriginal, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh - useful for CODA computations where the active part is extracted
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)

mask = generateBlankingMask(
    clac=clac, fsmesh=fsmeshOrig,
    offsets=offsets,
    meshID=meshID,
    localDir=localDirOut,
    offsetFromBC='BCWall',  # offsetFromBC='BCWall' or 'BCOverset' (by default)
    check=True
)

pyTree = extractPyTree(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, meshIDTarget=0) # extract background mesh
blankingObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, pyTree=pyTree) # MeshBlankingMap?

# This could be a time step loop...
for i in range(0, 1):

    # 1-update cell nature field with 0 (blanked) and 1 (active)
    blankingObj.computeBlanking(mask)

    # 2-remove blanked cells
    extractActiveSubMesh(dm, meshKeyOriginal, meshKeyActive)

    fsmeshOrig.ExportMeshTECPLOT(
        Filename=localDirOut+'blanking_mesh_%d.plt'%meshID,
        FileFormat='binary',
        PrefixDatasetName=True
    ) or FSError.PrintAndExit()

    fsmeshActive.ExportMeshTECPLOT(
        Filename=localDirOut+'blanked_mesh_%d.plt'%meshID,
        FileFormat='binary',
        PrefixDatasetName=True
    ) or FSError.PrintAndExit()