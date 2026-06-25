# Usage: kpython -n3 -t4 howToUseCassiopeeHoleCuttingTwoBodiesMeshIntersection.py
from FSDataManager import FSClac, FSError, FSDataManager

from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset, generateBlankingMask, extractPyTree, extractActiveSubMesh

"""
In this configuration the overset border of the child mesh of the cylinder
intersects the wall of the naca child mesh.
This is not acceptable because this would generate orphan cells (no donor found in CODA).
In this case the idea is to 'cut' the mesh that overlaps the wall of the other
mesh (in this case we cut the cylinder mesh), simply applying the same blanking
procedure that we usually (in the single body case) apply to the background mesh.
"""

offsets = [0.3, 0.1]

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
    meshKeyOriginal = 'naca'
    meshKeyActive = 'naca'
elif meshID == 2:
    meshFilename = localDirIn+'cylinder_large.h5'
    meshKeyOriginal = 'cyl_orig' # now the cylinder is also blanked
    meshKeyActive = 'cyl_active'

dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOriginal, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh - useful for CODA computations where the active part is extracted
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)

dictOfOffsets = {}
dictOfOffsets[1] = offsets[0]
dictOfOffsets[2] = offsets[1]

mask = generateBlankingMask(
    clac=clac, fsmesh=fsmeshOrig,
    dictOfOffsets=dictOfOffsets,
    meshID=meshID,
    localDir=localDirOut,
    offsetFromBC='BCWall',  # offsetFromBC='BCWall' or 'BCOverset' (by default)
    check=True
)
blankingObj0 = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, meshIDTarget=0)
blankingObj2 = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, meshIDTarget=2)
mask0 = mask.copy()
mask2 = mask.copy(); mask2.pop(2) # remove the mask around the cylinder (meshID=2)

# This could be a time step loop...
for i in range(0, 1):

    # 1-update cell nature field with 0 (blanked) and 1 (active)
    blankingObj0.computeBlanking(mask0)
    blankingObj2.computeBlanking(mask2)

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