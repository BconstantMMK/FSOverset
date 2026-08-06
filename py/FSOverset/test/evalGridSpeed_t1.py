# Usage: kpython -n1 -t4 evalGridSpeed_t1.py
import Generator.PyTree as G
import Converter.PyTree as C
import Converter.Internal as Internal
import Transform.PyTree as T
import RigidMotion.PyTree as R
import KCore.test as Ktest
from FSPlugins.test import testH5

from FSDataManager import FSError

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter
from FSOverset.FSOverset import initGridVelocity, evalGridSpeed

import numpy

testDir = Ktest.getDataFolderName()
check = False # comparison with RigidMotion.evalGridSpeed

motionDict = {'cart': {}}

def runTest(fsmesh, number):
    testFile = testDir+'/evalGridSpeed_t1_%d.h5'%number
    if not testFile: fsmesh.ExportMeshHDF5(Filename=testFile) or FSError.PrintAndExit()
    testH5(clac, fsmesh, number=1,
        checkCoordinates=True, coordsName='Coordinates',
        checkConnectivity=True, checkDatasets=True,
        rtol=0., atol=1.e-10,
        reference=testFile)

def printDiffRigidMotion(t, fsmesh, motionDict, number):
    R._setPrescribedMotion3(Internal.getNodeFromName1(t, 'cart'), 
        'rotation', 
        transl_speed=motionDict['cart']['transl_speed'], 
        axis_pnt=motionDict['cart']['axis_pnt'], 
        axis_vct=motionDict['cart']['axis_vct'], 
        omega=motionDict['cart']['angular_frq']
    )
    R._evalGridSpeed(t, 0.25)

    motion = Internal.getNodeByName3(t, 'Motion')
    VelocityX_pytree = Internal.getNodeFromName1(motion, 'VelocityX')[1]
    VelocityY_pytree = Internal.getNodeFromName1(motion, 'VelocityY')[1]
    VelocityZ_pytree = Internal.getNodeFromName1(motion, 'VelocityZ')[1]

    fs_gridVels = fsmesh.GetUnstructDataset('GridVelocity').GetValues()
    np_gridVels = numpy.array(fs_gridVels.Buffer(), copy=False)
    np_gridVels = numpy.reshape(np_gridVels, (-1,3))
    VelocityX_fsmesh = np_gridVels[:,0]
    VelocityY_fsmesh = np_gridVels[:,1]
    VelocityZ_fsmesh = np_gridVels[:,2]

    diffX = numpy.abs(VelocityX_pytree-VelocityX_fsmesh)
    diffY = numpy.abs(VelocityY_pytree-VelocityY_fsmesh)
    diffZ = numpy.abs(VelocityZ_pytree-VelocityZ_fsmesh)

    maxDiffX = numpy.max(diffX)
    maxDiffY = numpy.max(diffY)
    maxDiffZ = numpy.max(diffZ)
    
    print('\nCase %d: comparing FSOverset.evalGridSpeed with RigidMotion.evalGridSpeed'%number)
    print('maxDiffX/maxDiffY/maxDiffZ = %1.2e/%1.2e/%1.2e\n'%(maxDiffX, maxDiffY, maxDiffZ))

    return None

a = G.cartHexa((0,0,0), (0.1,0.1,0.1), (11,11,11))
T._rotate(a, (0.,0.,0.), (1,0,0), 45.)
T._rotate(a, (0.,0.,0.), (0,1,0), 45.)
t = C.newPyTree(['cart', a])

convObj = FSCGNSConverter(pyTree=t, flipYZAxes=False)
convObj.convert2FSDM()
clac = convObj.clac
fsmesh = convObj.fsmesh

# test 1 - initialize grid speed
initGridVelocity(fsmesh, 'cart', motionDict)
runTest(fsmesh, 1)

# test 2 - simple translation
motionDict['cart'] = {
    'transl_speed': [1.,0.,0.],
    'axis_pnt': [0.5, 0.5, 0.5],
    'axis_vct': [0.,1.,0.],
    'angular_frq': 0.
}
evalGridSpeed(fsmesh, 'cart', 0.25, motionDict)
if check: printDiffRigidMotion(t, fsmesh, motionDict, 2)
runTest(fsmesh, 2)

# test 3 - simple rotation
motionDict['cart'] = {
    'transl_speed': [0.,0.,0.],
    'axis_pnt': [0.5, 0.5, 0.5],
    'axis_vct': [0.,1.,0.],
    'angular_frq': 0.038
}
evalGridSpeed(fsmesh, 'cart', 0.25, motionDict)
if check: printDiffRigidMotion(t, fsmesh, motionDict, 3)
runTest(fsmesh, 3)

# test 4 - rotation + translation
motionDict['cart'] = {
    'transl_speed': [1.,0.,0.],
    'axis_pnt': [0.5, 0.5, 0.5],
    'axis_vct': [0.,1.,0.],
    'angular_frq': 0.038
}
evalGridSpeed(fsmesh, 'cart', 0.25, motionDict)
if check: printDiffRigidMotion(t, fsmesh, motionDict, 4)
runTest(fsmesh, 4)

# test 5 - oscillation
motionDict['cart'] = {
    'transl_speed': [0.,0.,0.],
    'axis_pnt': [0.5, 0.5, 0.5],
    'axis_vct': [0.,1.,0.],
    'angular_frq': 0.038,
    'mean_angle': -0.016,
    'ampl_angle': -2.51
}
evalGridSpeed(fsmesh, 'cart', 0.25, motionDict)
runTest(fsmesh, 5)