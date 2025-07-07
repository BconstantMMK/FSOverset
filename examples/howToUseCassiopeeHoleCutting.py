# Usage: kpython -n2 -t24 holeCutting.py
import os
import sys

from FSDataManager import FSClac, FSError, FSDataManager

from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset, holeMesh, backgroundMesh

offsets = [0.3]

globalClac = FSClac()  # by default, FSClac uses MPI_COMM_WORLD, i.e. all processes available
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 2:
    print("Must be run with at least 2 MPI processes!")
    sys.exit(os.EX_USAGE)

globalProcID = globalClac.GetProcID()
meshID = int(globalProcID // (nGlobalProcs / 2))

clac = FSClac()
globalClac.DivideIntoGroups(meshID, clac)

discParaDict = {}
wallBoundaryMarkers = []
if meshID == 0:
    meshFilename = "naca_background.h5"
    meshKeyOriginal = "back_orig"  # the original background mesh
    meshKeyActive = "back_active"  # the active part of the background mesh
    discParaDict['boundary treatments'] = [
        {
            'treatment type': 'BCFarfield',
            'boundary markers': [21, 22, 25, 26]
        },
        {
            'treatment type': 'BCSymmetryPlane',
            'boundary markers': [23, 24]
        }
    ]
else:
    meshFilename = "naca_curvi.h5"
    meshKeyOriginal = "child"  # the original child mesh
    meshKeyActive = "child"
    wallBoundaryMarkers_childmesh = [3]
    discParaDict['boundary treatments'] = [
        {
            'treatment type': 'BCWallViscousAdiabatic',
            'boundary markers': wallBoundaryMarkers_childmesh
        },
        {
            'treatment type': 'BCOverset',
            'boundary markers': [2]
        },
        {
            'treatment type': 'BCSymmetryPlane',
            'boundary markers': [1]
        }
    ]
    wallBoundaryMarkers = wallBoundaryMarkers_childmesh

fsDMObj = FSDataManager(globalClac)
fsmeshOrig = fsDMObj.GetMesh(meshKeyOriginal, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh (not used here but useful for CODA computations where the active part is extracted)
fsmeshActive = fsDMObj.GetMesh(meshKeyActive, clac, True)

print("meshID, globalProcID, meshKeyOriginal:", meshID, globalProcID, meshKeyOriginal)
fsmeshOrig.PrintInfo()

t_hole = holeMesh(
    clac=clac, fsmesh=fsmeshOrig,
    paraDict=discParaDict,
    offsets=offsets,
    meshID=meshID,
    offsetFromBC="BCWall"  # offsetFromBC="BCWall" or "BCOverset" (by default)
)
t_bg = backgroundMesh(clac=clac, fsmesh=fsmeshOrig, meshID=meshID)
blankingObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, pyTree=t_bg)


# This could be a time step loop...
for i in range(0, 1):

    # Maybe transform computational meshes and hole definition meshes first...
    blankingObj.computeBlanking(t_hole)

    fsmeshOrig.ExportMeshTECPLOT(
        Filename=f"blanking_mesh_{meshID:d}.plt",
        FileFormat="ASCII",
        PrefixDatasetName=True
    ) or FSError.PrintAndExit()

    # In case of CODA: extraction of active mesh parts, computation, solution transfer to original complete meshes...
