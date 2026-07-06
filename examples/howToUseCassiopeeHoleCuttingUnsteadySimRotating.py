from FSDataManager import FSClac, FSError, FSDataLog, FSDataManager

from CODA import DiscretizationFactory, TimeIntegrationFactory
from CODA import StopNumIterations, StopRelativeReduction
from CODA import MonitorTabular, MonitorSelection, MonitorIntegrals, Monitor
from CODA.CODAHelpers import BuildDiscretizationParameterTrees, BuildTimeIntegrationParameterTrees

from FSOverset.FSOverset import FSOverset, generateBlankingMask, extractPyTree, extractActiveSubMesh, copySolution, generateDiscParasFromMesh
from FSOverset.FSOverset import initGridVelocity, copyGrid2GridInit, evalPosition, evalGridSpeed, getWallBoundaryMarkers, display
from FSCGNSConverter.FSCGNSConverter import buildMeshOps

import math

# overset settings
offsetFromBC = 'BCWall'

# mesh settings
localDirIn = 'INPUT/'
localDirOut = 'OUTPUT/TEST_ROTATING/'

# flow settings
Mach, gamma, chord = 0.2, 1.4, 1.0
k = 0.0814 # reduced frequency
tx, ty, tz = (0., 0., 0.) # translation vector
cx, cy, cz = (0.25, 0., 0.) # center of rotation
kx, ky, kz = (0., 1., 0.) # axis vector

Uinf = Mach * math.sqrt(gamma) # freestream speed
omega = 2 * k * Uinf / chord # angular frequency
period = 2 * math.pi / omega # oscillatory period

# solver settings
TSOP = 200 # time steps per oscillatory period
nperiod = 1 # number of oscillatory periods
targetResidualReduction = 1.0e-8
maximumNumberOfIterations = 200

time_step = period / TSOP
niter = nperiod * TSOP
time = 0

motionDict = {
    'transl_speed': [tx, ty, tz],
    'axis_pnt': [cx, cy, cz],
    'axis_vct': [kx, ky, kz],
    'angular_frq': omega
}

displayDict = {
    'isoScales': {'State.Density': ['State.Density', 25, 0.92, 1.02]},
    'xlim': [-1.5, 2.5],
    'ylim': [-1.0, 1.0],
    'zplane': 0.0,
    'mpl': False
}

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
            "Mach": Mach,
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

innerTimeIntegrationParaDict = {
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
outerTimeIntegrationParaDict = {
    "time integration method": "[(E)S]DIRK",
    "time information logging name": "PhysicalTime",
    "Butcher tableau": {"identifier": "Alexander22"},
    "time step": {
        "type": "constant",
        "size": time_step,
    },
}
## ====================================
## Set up blanking data - user defined
## ====================================
offsetDict={}
offsetDict[1]=0.3

blankingDict = {}
blankingDict[0]=[1]

#===============================================
globalClac = FSClac()  # by default, FSClac uses MPI_COMM_WORLD, i.e. all processes available
globalProcID = globalClac.GetProcID()
nGlobalProcs = globalClac.GetNProcs()

if nGlobalProcs < 2:
    raise ValueError("howToUseCassiopeeHoleCuttingUnsteadySim must be run with at least 2 MPI processes.")

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

master = localClac.GetProcID() == 0
clacMaster  = FSClac()
globalClac.DivideIntoGroups(master, clacMaster)

## ====================================
## Import Mesh
## ====================================
if meshID == 0:
    meshFilename = localDirIn+'background.h5'
    meshKeyOriginal = 'back_orig'  # the original background mesh
    meshKeyActive = 'back_active'  # the active part of the background mesh
else:
    meshFilename = localDirIn+'naca.h5'
    meshKeyOriginal = 'naca_orig'
    meshKeyActive = 'naca_active'


# MANDATORY to set to 'none' for non-blanked meshes for extractActiveSubMesh to work properly
if meshID not in blankingDict:
   meshKeyOriginal = 'none'
   meshKeyActive = 'none'

## ====================================
## Create Data Manager & set Original/Active
## ====================================    
dm = FSDataManager(globalClac)

fsmeshOriginal = dm.GetMesh(meshKeyOriginal, localClac, True)
meshOps = buildMeshOps(meshFilename, preserveCellStacks=True, verbose=True)
fsmeshOriginal.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh - useful for CODA computations where the active part is extracted
fsmeshActive = dm.GetMesh(meshKeyActive, localClac, True)

#blanking objs creation
blankingMaskDict = generateBlankingMask(
    clac=localClac, fsmesh=fsmeshOriginal,
    offsetDict=offsetDict,
    meshID=meshID,
    localDir=localDirOut,
    offsetFromBC=offsetFromBC,
    check=False)
blankingObj = FSOverset(clac=localClac, fsmesh=fsmeshOriginal, meshID=meshID, blankingDict=blankingDict)

# il faut le mettre 1 fois pour initialiser le fsmeshactive pour creer le local numbering
blankingObj.computeBlanking(blankingMaskDict=blankingMaskDict)
extractActiveSubMesh(dm, meshKeyOriginal, meshKeyActive) 

## ====================================
## Set up CODA Dicts & Settings
## ====================================
fsmeshActive.CreateLocalNumbering() # mandatory for Cmpi.size > 2

discParaDict = generateDiscParasFromMesh(fsmeshActive, discParaDict)

discSelectionParas, discParas = BuildDiscretizationParameterTrees(discSelectionParaDict, discParaDict)
discFactory = DiscretizationFactory.GetSingleton()
disc = discFactory.Create(discSelectionParas, globalClac, fsmeshActive, discParas)

timeIntegrationParasAllLevels = BuildTimeIntegrationParameterTrees([outerTimeIntegrationParaDict, innerTimeIntegrationParaDict])
timeIntegrationParasOuter = timeIntegrationParasAllLevels[0]
timeIntegrationParasInner = timeIntegrationParasAllLevels[1]
timeFactory = TimeIntegrationFactory.GetSingleton()
timeIntegration = timeFactory.Create(disc, timeIntegrationParasAllLevels)

residualNames = ['DensityResidual', 'MomentumResidual', 'EnergyStagnationDensityResidual']
monitorVariables = ['CFL'] + ['%sReduction'%res for res in residualNames]
reductionCallback = StopRelativeReduction(residualNames, targetResidualReduction)

wallMarkers = getWallBoundaryMarkers(fsmesh=fsmeshActive)
monitorIntegralsCallbacks = MonitorIntegrals(
    'boundary',
    [
        {
            'varnames': ['CoefDrag', 'CoefLift', 'CoefMomentY'],
            'markers': wallMarkers,
            'levels': [0], # outerIntegration
            'key': 'AerodynamicCoefficients',
        },
    ],
    globalClac=globalClac,
    meshLocalClac=localClac,
    meshClacMaster=clacMaster,
    monitorPeriod=1,
    timeLoggingName='PhysicalTime',
    multiMesh=True
)

iterationCallbacksInner = (reductionCallback | StopNumIterations(maximumNumberOfIterations)) + MonitorTabular(
    globalClac,
    disc.GetStateVariableNames(),
    timeIntegrationParasInner['state backup controller'],
    # monitorIntegralsCallbacks=[monitorIntegralsCallbacks],
    monitorVariables=monitorVariables,
    monitorSelection=MonitorSelection(monitorWallClockTime=True),
    monitorPeriod=1,
    includeTimeColumn=True,
)

iterationCallbacksOuter = StopNumIterations(1) + monitorIntegralsCallbacks

dataLog = FSDataLog(globalClac)

# fsmesh in relative motion
if meshID != 0:
    initGridVelocity(fsmeshOriginal)
    copyGrid2GridInit(fsmeshOriginal, blankingMaskDict)
else:
    copyGrid2GridInit(None, blankingMaskDict)

## ====================================
## Compute loop
## ====================================

for i in range(niter):
    time += time_step

    # update grid coordinates and grid velocities
    if meshID != 0:
        evalPosition(fsmeshOriginal, blankingMaskDict, time, motionDict=motionDict)
        evalGridSpeed(fsmeshOriginal, time, motionDict=motionDict)
    else:
        evalPosition(None, blankingMaskDict, time, motionDict=motionDict)

    # update blanking    
    blankingObj.computeBlanking(blankingMaskDict=blankingMaskDict)
    extractActiveSubMesh(dm, meshKeyOriginal, meshKeyActive)

    # update CODA settings
    fsmeshActive.HasLocalNumbering() or fsmeshActive.CreateLocalNumbering()
    disc = discFactory.Create(discSelectionParas, globalClac, fsmeshActive, discParas)
    timeIntegration = timeFactory.Create(disc, timeIntegrationParasAllLevels)
    if i == 0:
        state = disc.CreateZeroFieldVector()
        disc.InitializeFieldVector({'type': 'free stream'}, state)
    else:
        state = disc.CreateZeroFieldVector()
        state.ImportFromFSMesh(disc.GetMeshInterface(), fsmeshActive, 'State') or FSError.PrintAndExit()
    
    # solution process
    status = timeIntegration.Iterate([iterationCallbacksOuter, iterationCallbacksInner], state, dataLog)

    # copy solution to active grids
    state.ExportToFSMesh(disc.GetMeshInterface(), fsmeshActive, 'State') or FSError.PrintAndExit()

    # copy solution to original grids
    copySolution(dm, meshKeyOriginal, meshKeyActive)

    # export image with Cassiopee
    #display(globalClac, fsmeshActive, meshID, ['State.Density'], it=i+1, displayDict=displayDict, localDir=localDirOut, saveTree=True)

# export flow solution
fsmeshActive.ExportMeshHDF5(HDF5Filename=localDirOut+'fsmeshActive_meshID%d_iter%d.h5'%(meshID, i+1), FilePerProcess=False) or FSError.PrintAndExit()
fsmeshOriginal.ExportMeshHDF5(HDF5Filename=localDirOut+'fsmeshOriginal_meshID%d_iter%d.h5'%(meshID, i+1), FilePerProcess=False) or FSError.PrintAndExit()

# export data logs
dataLog.ExportDataXML(localDirOut+'datalog.xml') or FSError.PrintAndExit()

# convergence history output
dataLog.ExportDataTECPLOT(localDirOut+'datalog.dat') or FSError.PrintAndExit()