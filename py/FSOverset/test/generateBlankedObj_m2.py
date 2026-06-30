# Usage: kpython -n3 -t4 generateBlankedObj_m2.py
from FSDataManager import FSClac, FSError, FSDataManager

from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset
import Converter.PyTree as C
import KCore.test as Ktest
import Converter.Mpi as Cmpi
rank = Cmpi.rank

offsets = [0.3,0.1]
localDirIn = './INPUT/'

globalClac = FSClac()  # by default, FSClac uses MPI_COMM_WORLD, i.e. all processes available
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 3:
    raise ValueError("howToUseCassiopeeHoleCutting must be run with at least 2 MPI processes.")

globalProcID = globalClac.GetProcID()
meshID = int(globalProcID // (nGlobalProcs / 3))

clac = FSClac()
globalClac.DivideIntoGroups(meshID, clac)

if meshID == 0:
    meshFilename = localDirIn+'background.h5'
    meshKeyOrig = 'back_orig'  # the Orig background mesh
    meshKeyActive = 'back_active'  # the active part of the background mesh
elif meshID == 1:
    meshFilename = localDirIn+'naca.h5'
    meshKeyOrig = 'naca'
    meshKeyActive = 'naca'
elif meshID == 2:
    meshFilename = localDirIn+'cylinder_large.h5'
    meshKeyOrig = 'cyl_orig' # now the cylinder is also blanked
    meshKeyActive = 'cyl_active'

dictOfOffsets = {}
dictOfOffsets[1] = offsets[0]
dictOfOffsets[2] = offsets[1]
dictOfOffsets[0] = 0.01 # there is no BCWall so it won t be taken into account

dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOrig, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh - useful for CODA computations where the active part is extracted
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)


for meshIDTarget in [0,2]:
    #pyTree0 = extractPyTree(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, meshIDTarget=meshIDTarget) # extract background mesh
    #blankingObj0 = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, meshIDTarget=meshIDTarget, pyTree=pyTree0)
    blankedObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, meshIDTarget=meshIDTarget)
    t = blankedObj.pyTree        

    if meshIDTarget==meshID:
        C.convertPyTree2File(t, 'blankedObj_meshID%d_%d.cgns'%(meshID, rank))
    if t is None: t = C.newPyTree(["DUMMY_%d_%d"%(meshIDTarget, rank)])
    notest = rank+meshIDTarget*nGlobalProcs
    Ktest.testT(t, notest)
