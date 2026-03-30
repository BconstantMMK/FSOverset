# Usage: kpython -n2 -t4 howToUseCassiopeeHoleCutting.py
from FSDataManager import FSClac, FSError, FSDataManager

from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset, generateBlankingMask, extractPyTree, extractActiveSubMesh

offsets = [0.3]
localDir = './OUTPUT/TEST1/'

globalClac = FSClac()  # by default, FSClac uses MPI_COMM_WORLD, i.e. all processes available
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 2:
    raise ValueError("howToUseCassiopeeHoleCutting must be run with at least 2 MPI processes.")

globalProcID = globalClac.GetProcID()
meshID = int(globalProcID // (nGlobalProcs / 2))

clac = FSClac()
globalClac.DivideIntoGroups(meshID, clac)

if meshID == 0:
    meshFilename = 'naca_background.h5'
    meshKeyOrig = 'back_orig'  # the original background mesh
    meshKeyActive = 'back_active'  # the active part of the background mesh
else:
    meshFilename = 'naca_curvi.h5'
    meshKeyOrig = 'naca'
    meshKeyActive = 'naca'

dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOrig, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh - useful for CODA computations where the active part is extracted)
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)

mask = generateBlankingMask(
    clac=clac, fsmesh=fsmeshOrig,
    offsets=offsets,
    meshID=meshID,
    localDir=localDir,
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
    extractActiveSubMesh(dm, meshKeyOrig, meshKeyActive)

    fsmeshOrig.ExportMeshTECPLOT(
        Filename=localDir+'blanking_mesh_%d.plt'%meshID,
        FileFormat='binary',
        PrefixDatasetName=True
    ) or FSError.PrintAndExit()

    fsmeshActive.ExportMeshTECPLOT(
        Filename=localDir+'blanked_mesh_%d.plt'%meshID,
        FileFormat='binary',
        PrefixDatasetName=True
    ) or FSError.PrintAndExit()