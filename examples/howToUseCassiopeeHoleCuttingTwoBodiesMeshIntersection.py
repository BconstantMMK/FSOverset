# Usage: kpython -n3 -t16 holeCuttingTwoBodiesMeshIntersection.py
import os
import sys

from FSDataManager import FSClac, FSError, FSDataManager

from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset, holeMesh, backgroundMesh, childMesh

"""
In this configuration the overset border of the child mesh of the cylinder
intersects the wall of the naca child mesh.
This is not acceptable because this would generate orphan cells (no donor found in CODA).
In this case the idea is to "cut" the mesh that overlaps the wall of the other
mesh (in this case we cut the cylinder mesh), simply applying the same blanking
procedure that we usually (in the single body case) apply to the background mesh.
"""

offsets = [0.3, 0.1]

globalClac = FSClac()  # by default, FSClac uses MPI_COMM_WORLD, i.e. all processes available
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 3:
    print("Must be run with at least 3 MPI processes!")
    sys.exit(os.EX_USAGE)

globalProcID = globalClac.GetProcID()
meshID = int(globalProcID // (nGlobalProcs / 3))

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
elif meshID == 1:
    meshFilename = "naca_curvi.h5"
    meshKeyOriginal = "child"  # the original child mesh
    meshKeyActive = "child"
    wallBoundaryMarkers_childmeshnaca = [3]
    discParaDict['boundary treatments'] = [
        {
            'treatment type': 'BCWallViscousAdiabatic',
            'boundary markers': wallBoundaryMarkers_childmeshnaca
        },
        {
            'treatment type': 'BCOverset',
            'boundary markers': [2]
        },
        {
            'treatment type': 'BCSymmetryPlane',
            'boundary markers': [1]
        },
    ]
    wallBoundaryMarkers = wallBoundaryMarkers_childmeshnaca
elif meshID == 2:
    meshFilename = "cyl_curvi_bigger.h5"
    meshKeyOriginal = "back_orig" # This has changed!! Now it is a "background mesh" because the cylinder will be blanked as well, and in a CODA computation, the blanked zone of the cylinder (overlapping) will be removed
    meshKeyActive = "back_active"
    wallBoundaryMarkers_childmeshcyl = [3]
    discParaDict['boundary treatments'] = [
        {
            'treatment type': 'BCWallViscousAdiabatic',
            'boundary markers': wallBoundaryMarkers_childmeshcyl
        },
        {
            'treatment type': 'BCOverset',
            'boundary markers': [2]
        },
        {
            'treatment type': 'BCSymmetryPlane',
            'boundary markers': [1]
        },
    ]
    wallBoundaryMarkers = wallBoundaryMarkers_childmeshcyl

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
t_curviCyl = childMesh(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, meshIDChildMeshToBlank=2)  # meshID = 2 is the cylinder

blankingObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, pyTree=t_bg)
blankingObj2 = FSOverset(clac=clac, fsmesh=fsmeshOrig, pyTree=t_curviCyl)

# use the same t_hole that we have used to blank the mesh with meshID=0 (background),
# to blank the cylinder mesh with meshID = 2
t_hole2 = t_hole.copy()
t_hole2.pop(2)  # meshID = 1 is the naca, meshID=2 is the cylinder. Now we remove the hole computed around the cylinder and we keep the hole computed around the naca

# This could be a time step loop...
for i in range(0, 1):

    # Maybe transform computational meshes and hole definition meshes first...
    blankingObj.computeBlanking(t_hole)
    blankingObj2.computeBlanking(t_hole2)

    fsmeshOrig.ExportMeshTECPLOT(
        Filename=f"blanking_mesh_{meshID:d}.plt",
        FileFormat="ASCII",
        PrefixDatasetName=True
    ) or FSError.PrintAndExit()

    # In case of CODA: extraction of active mesh parts, computation, solution transfer to original complete meshes...
