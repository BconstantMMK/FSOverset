# Usage: kpython -n2 -t4 evalGridSpeed_m1.py
from FSDataManager import FSError, FSDataManager
from FSCGNSConverter.FSCGNSConverter import buildMeshOps
from FSOversetMotion.FSOversetMotion import initGridVelocity, evalGridSpeed, getClacInfo, getMeshKeys
from FSPlugins.test import testH5
import KCore.test as Ktest

localDirIn = './INPUT/'

meshDict = {
    'background': {'meshFilename': localDirIn+'background.h5', 'meshProcessorWeight': 1.},
    'naca': {'meshFilename': localDirIn+'naca.h5', 'meshProcessorWeight': 1.}
}
blankingDict = {
    'background': ['naca']
}
motionDict = {
    'naca': {
        'transl_speed': [0.,0.,0.],
        'axis_pnt': [0.25,0.,0.],
        'axis_vct': [0.,1.,0.],
        'angular_frq': 0.038
    }
}

# Get clacs
meshKey, meshColor, clac, globalClac, masterClac = getClacInfo(meshDict)
meshFilename = meshDict[meshKey]['meshFilename']
meshKeyActive, meshKeyOrig = getMeshKeys(meshKey, blankingDict)

# Get orig mesh
dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOrig, clac, True)
meshOps = buildMeshOps(meshFilename, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Init grid vel
initGridVelocity(fsmeshOrig, meshKey, motionDict)

# Eval grid vel at time = 0.25
evalGridSpeed(fsmeshOrig, meshKey, 0.25, motionDict)

# test
testDir = Ktest.getDataFolderName()
testFile = testDir+'/evalGridSpeed_m1_%d.h5'%globalClac.GetProcID()
testH5(clac, fsmeshOrig,
       coordsName='Coordinates',
       rtol=0., atol=1.e-10,
       reference=testFile)