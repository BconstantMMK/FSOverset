# Usage: kpython -n2 -t4 howToUseCassiopeeHoleCutting.py
from FSDataManager import FSClac, FSError, FSDataManager

from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset, generateBlankingMask, extractActiveSubMesh

localDirIn = './INPUT/'
localDirOut = './OUTPUT/TEST1/'

globalClac = FSClac()  # by default, FSClac uses MPI_COMM_WORLD, i.e. all processes available
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 2:
    raise ValueError("howToUseCassiopeeHoleCutting must be run with at least 2 MPI processes.")

globalProcID = globalClac.GetProcID()
meshID = int(globalProcID // (nGlobalProcs / 2))

clac = FSClac()
globalClac.DivideIntoGroups(meshID, clac)

if meshID == 0:
    meshFilename = localDirIn+'background.h5'
    meshKeyOrig = 'back_orig'  # the original background mesh
    meshKeyActive = 'back_active'  # the active part of the background mesh
else:
    meshFilename = localDirIn+'naca.h5'
    meshKeyOrig = 'naca'
    meshKeyActive = 'naca'


dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOrig, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh - useful for CODA computations where the active part is extracted
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)

dictOfOffsets={}
dictOfOffsets[1] = 0.3
mask = generateBlankingMask(
    clac=clac, fsmesh=fsmeshOrig,
    dictOfOffsets=dictOfOffsets,
    meshID=meshID,
    localDir=localDirOut,
    offsetFromBC='BCWall',  
    check=False)

dictOfBlanking = {
    0:[1]
}
blankingObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, dictOfBlanking=dictOfBlanking)

# This could be a time step loop...
for i in range(0, 1):

    # 1-update cell nature field with 0 (blanked) and 1 (active)
    blankingObj.computeBlanking(dictOfMasks=mask)

    # 2-remove blanked cells
    extractActiveSubMesh(dm, meshKeyOrig, meshKeyActive)

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