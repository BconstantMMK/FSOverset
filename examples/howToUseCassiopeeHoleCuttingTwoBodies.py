# Usage: kpython -n3 -t16 holeCuttingTwoBodies.py
import os
import sys

from FSDataManager import FSClac, FSError, FSDataManager

from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset, holeMesh, backgroundMesh

"""
The following three cases have the SAME treatment, both in the blanking process and in CODA.
In case the blanked area of a child mesh intersects the WALL of the other child mesh,
refer to example howToUseCassiopeeHoleCuttingTwoBodiesDoubleBlanking.py.
"""

offsets = [0.03,0.05] # case in which there is no intersection, neither of the blanked areas of the two child meshes nor of a child mesh overset border with blanked area of the other child mesh
#offsets = [0.2,0.05] # case in which the blanked areas of the two child meshes do not intersect, but the cylinder child mesh overset border intersects the blanked area of the naca child mesh
#offsets = [0.3,0.5]  # case in which the blanked areas of naca and cylinder child mesh intersect


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
    meshKeyOriginal = "child" # the original child mesh
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
        }
    ]
    wallBoundaryMarkers = wallBoundaryMarkers_childmeshnaca
else:
    meshFilename = "cyl_curvi.h5"
    meshKeyOriginal = "child" # the original child mesh
    meshKeyActive = "child"
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
