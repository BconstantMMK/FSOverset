from FSDataManager import FSClac, FSError, FSDataLog, FSDataManager

from CODA import DiscretizationFactory, TimeIntegrationFactory
from CODA import StopNumIterations, StopRelativeReduction
from CODA import MonitorTabular, MonitorSelection
from CODA.CODAHelpers import BuildDiscretizationParameterTrees, BuildTimeIntegrationParameterTrees

from FSOverset.FSOverset import FSOverset, generateBlankingMask, extractPyTree, extractActiveSubMesh, copySolution, generateDiscParasFromMesh
from FSCGNSConverter.FSCGNSConverter import buildMeshOps

import math

# overset settings
offsets = [0.3]
offsetFromBC = 'BCWall'

# mesh settings
localDirIn = 'INPUT/'
localDirOut = 'OUTPUT/TEST_RANS/'

# solver settings
targetResidualReduction = 1.0e-8
maximumNumberOfIterations = 200

discSelectionParaDict = {
    "PDE" : "Euler",
    "spatial scheme" : "FV",
    "convection scheme" : "Roe upwinding",
    "order" : 2,
}
discParaDict = {
    "testing": {
        "willfully ignore excessive load imbalance among domains w.r.t. the number of faces": True,
    },
    "reference state": {
        "flow speed specification": {
            "type": "Mach number based",
            "Mach": 0.755,
        },
        "flow direction specification": {
            "type": "aerodynamic flow angles",
            "angle of attack": 0.0,
        },
    },
    "preprocessing" : {
        "maximum relative surface integral" : 1e-13,
        "non-local boundary treatments" : {
            "epsilon for in-element check" : 1e-13,
        },
    },
    "reconstruction": {
        "face gradient augmentation": "cell-to-face",
        "face gradient augmentation for the Jacobian matrix": "cell-to-face",
        "type": "limited linear",
        "gradient limiter": {
            "type": "full limiting",
        },
    },
    "boundary integral quantities": {
        "Coef_Area": 1.0,
        "Moment_Center": [0.25, 0.0, 0.0],
        "Coef_Length": 1.0,
    },
}
timeIntegrationParaDict = {
    "time integration method": "linearized implicit Euler",
    "time step": {
        "type": "local",
        "CFL": {
            "type": "SER ramp max refmax",
            "initial CFL number": 1.0,
            "maximum CFL number": 1000,
            "SER exponent": 0.5,
        },
    }
}

globalClac = FSClac()  # by default, FSClac uses MPI_COMM_WORLD, i.e. all processes available
globalProcID = globalClac.GetProcID()
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 2:
    raise ValueError("howToUseCassiopeeHoleCuttingSteadySim must be run with at least 2 MPI processes.")

## ====================================
## Set local clacs
## ====================================
weightBackground = 4.0
weightAirfoil = 1.0
weightSum = weightBackground + weightAirfoil

nProcsAirfoil = math.ceil(nGlobalProcs * weightAirfoil / weightSum)
nProcsBackground = nGlobalProcs - nProcsAirfoil

meshID = 0 if globalProcID in range(0, nProcsBackground) else 1
    
localClac = FSClac()
globalClac.DivideIntoGroups(meshID, localClac)

## ====================================
## Import Mesh
## ====================================
if meshID == 0:
    meshFilename = localDirIn+'background.h5'
    meshKeyOriginal = 'back_orig'  # the original background mesh
    meshKeyActive = 'back_active'  # the active part of the background mesh
else:
    meshFilename = localDirIn+'naca.h5'
    meshKeyOriginal = 'naca'
    meshKeyActive = 'naca'

## ====================================
## Create Data Manager & set Original/Active
## ====================================    
dm = FSDataManager(globalClac)

fsmeshOriginal = dm.GetMesh(meshKeyOriginal, localClac, True)
fsmeshActive = dm.GetMesh(meshKeyActive, localClac, True)

meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=True)
fsmeshOriginal.DoOps(meshOps) or FSError.PrintAndExit()

if meshID != 0: 
    meshKeyOriginal = 'none'
    meshKeyActive = 'none'

## ====================================
## Set up blanking & overset
## ====================================
mask = generateBlankingMask(
    clac=localClac, fsmesh=fsmeshOriginal,
    offsets=offsets,
    meshID=meshID,
    localDir=localDirOut,
    offsetFromBC=offsetFromBC,
    check=True
)

pyTree = extractPyTree(clac=localClac, fsmesh=fsmeshOriginal, meshID=meshID, meshIDTarget=0) # extract background mesh
blankingObj = FSOverset(clac=localClac, fsmesh=fsmeshOriginal, pyTree=pyTree)
blankingObj.computeBlanking(mask)

extractActiveSubMesh(dm, meshKeyOriginal, meshKeyActive)

## ====================================
## Set up CODA Dics & Settings
## ====================================
fsmeshActive.CreateLocalNumbering() # mandatory for Cmpi.size > 2

discParaDict = generateDiscParasFromMesh(fsmeshActive, discParaDict)

discSelectionParas, discParas = BuildDiscretizationParameterTrees(discSelectionParaDict, discParaDict)
disc = DiscretizationFactory.GetSingleton().Create(discSelectionParas, globalClac, fsmeshActive, discParas)

timeIntegrationParasAllLevels = BuildTimeIntegrationParameterTrees([timeIntegrationParaDict])
timeIntegrationParas = timeIntegrationParasAllLevels[0]
timeIntegration = TimeIntegrationFactory.GetSingleton().Create(disc, timeIntegrationParasAllLevels)

state = disc.CreateZeroFieldVector()
disc.InitializeFieldVector({'type': 'free stream'}, state)

residualNames = ['DensityResidual', 'MomentumResidual', 'EnergyStagnationDensityResidual']
monitorVariables = ['CFL'] + ['%sReduction'%res for res in residualNames]
reductionCallback = StopRelativeReduction(residualNames, targetResidualReduction)

iterationCallbacks = (reductionCallback | StopNumIterations(maximumNumberOfIterations)) + MonitorTabular(
    globalClac,
    disc.GetStateVariableNames(),
    timeIntegrationParas['state backup controller'],
    monitorVariables=monitorVariables,
    monitorSelection=MonitorSelection(monitorWallClockTime=False),
    monitorPeriod=1,
)

dataLog = FSDataLog(globalClac)

## ====================================
## Compute loop
## ====================================
# solution process
status = timeIntegration.Iterate([iterationCallbacks], state, dataLog)

# copy solution to active grids
state.ExportToFSMesh(disc.GetMeshInterface(), fsmeshActive, 'State') or FSError.PrintAndExit()

# copy solution to original grids
copySolution(dm, meshKeyOriginal, meshKeyActive)

# export convergence history
dataLog.ExportDataTECPLOT(localDirOut+'monitor.dat', 'l2-norms') or FSError.PrintAndExit()

# export flow solution
fsmeshActive.ExportMeshHDF5(HDF5Filename=localDirOut+'fsmeshActive_meshID%d.h5'%(meshID), FilePerProcess=False) or FSError.PrintAndExit()
fsmeshOriginal.ExportMeshHDF5(HDF5Filename=localDirOut+'fsmeshOriginal_meshID%d.h5'%(meshID), FilePerProcess=False) or FSError.PrintAndExit()