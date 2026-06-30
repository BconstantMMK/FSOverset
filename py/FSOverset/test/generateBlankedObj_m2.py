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
    raise ValueError("howToUseCassiopeeHoleCutting must be run with at least 3 MPI processes.")

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

dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOrig, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh - useful for CODA computations where the active part is extracted
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)

dictOfBlanking={}
dictOfBlanking[0] = [1,2]
dictOfBlanking[2] = [1]

blankedObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, dictOfBlanking=dictOfBlanking)
t = blankedObj.pyTree        
if t is None: t = C.newPyTree(["DUMMY_%d"%meshID])
if meshID == globalProcID:
    Ktest.testT(t,meshID)
