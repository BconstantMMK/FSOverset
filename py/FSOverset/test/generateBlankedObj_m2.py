# Usage: kpython -n3 -t4 generateBlankedObj_m2.py
from FSDataManager import FSClac, FSError, FSDataManager
from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOverset.FSOverset import FSOverset
import Converter.PyTree as C
import KCore.test as Ktest

localDirIn = './INPUT/'

dictOfBlanking = {}
dictOfBlanking[0] = [1,2]
dictOfBlanking[2] = [1]

globalClac = FSClac()  # by default, FSClac uses MPI_COMM_WORLD, i.e. all processes available
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 3:
    raise ValueError('generateBlankedObj_m2.py requires 2 processes at least.')

globalProcID = globalClac.GetProcID()
meshID = int(globalProcID // (nGlobalProcs / 3))

clac = FSClac()
globalClac.DivideIntoGroups(meshID, clac)

if meshID == 0:
    meshFilename = localDirIn+'background.h5'
    meshKeyOrig = 'back_orig'
    meshKeyActive = 'back_active'
elif meshID == 1:
    meshFilename = localDirIn+'naca.h5'
    meshKeyOrig = 'naca_orig'
    meshKeyActive = 'naca_active'
else:
    meshFilename = localDirIn+'cylinder_large.h5'
    meshKeyOrig = 'cyl_orig'
    meshKeyActive = 'cyl_active'

dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOrig, clac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Initialize FSOverset class for every meshID
blankedObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshID=meshID, dictOfBlanking=dictOfBlanking)

# test
t = blankedObj.pyTree        
if t is None: t = C.newPyTree(['DUMMY_%d'%meshID])
Ktest.testT(t,globalProcID)
