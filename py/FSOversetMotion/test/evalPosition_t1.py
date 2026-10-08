# Usage: kpython -n1 -t4 evalPosition_t1.py
import Generator.PyTree as G
import Converter.PyTree as C
import Converter.Internal as Internal
import Transform.PyTree as T
import RigidMotion.PyTree as R
import KCore.test as Ktest
from FSPlugins.test import testH5

from FSDataManager import FSError

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter
from FSOversetMotion.FSOversetMotion import copyGrid2GridInit, evalPosition

import numpy

testDir = Ktest.getDataFolderName()
check = False # comparison with RigidMotion.evalPosition

motionDict = {'cart': {}}

def runTest(fsmesh, number):
    testFile = testDir+'/evalPosition_t1_%d.h5'%number
    testH5(clac, fsmesh,
           coordsName='Coordinates',
           rtol=0., atol=1.e-10,
           reference=testFile)

def printDiffRigidMotion(t, fsmesh, motionDict, number):
    if motionDict is not None:
        R._setPrescribedMotion3(Internal.getNodeFromName1(t, 'cart'), 
            'rotation', 
            transl_speed=motionDict['cart']['transl_speed'], 
            axis_pnt=motionDict['cart']['axis_pnt'], 
            axis_vct=motionDict['cart']['axis_vct'], 
            omega=motionDict['cart']['angular_frq']
        )
        R._evalPosition(t, 0.25)

    coords = Internal.getNodeByName(t, Internal.__GridCoordinates__)
    CoordX_pytree = Internal.getNodeFromName1(coords, 'CoordinateX')[1]
    CoordY_pytree = Internal.getNodeFromName1(coords, 'CoordinateY')[1]
    CoordZ_pytree = Internal.getNodeFromName1(coords, 'CoordinateZ')[1]

    fs_gridCoords = fsmesh.GetUnstructDataset('Coordinates').GetValues()
    np_gridCoords = numpy.array(fs_gridCoords.Buffer(), copy=False)
    CoordX_fsmesh = np_gridCoords[:,0]
    CoordY_fsmesh = np_gridCoords[:,1]
    CoordZ_fsmesh = np_gridCoords[:,2]

    diffX = numpy.abs(CoordX_pytree-CoordX_fsmesh)
    diffY = numpy.abs(CoordY_pytree-CoordY_fsmesh)
    diffZ = numpy.abs(CoordZ_pytree-CoordZ_fsmesh)
    
    print('\nCase %d: comparing FSOversetMotion.evalGridSpeed with RigidMotion.evalGridSpeed'%number)
    print('maxDiffX/maxDiffY/maxDiffZ = %1.2e/%1.2e/%1.2e\n'%(numpy.max(diffX), numpy.max(diffY), numpy.max(diffZ)))

    return None

a = G.cartHexa((0,0,0), (0.1,0.1,0.1), (11,11,11))
T._rotate(a, (0.,0.,0.), (1,0,0), 45.)
T._rotate(a, (0.,0.,0.), (0,1,0), 45.)
t = C.newPyTree(['cart', a])
R._copyGrid2GridInit(t, mode=1)

convObj = FSCGNSConverter(pyTree=t, flipYZAxes=False)
convObj.convert2FSDM()
clac = convObj.clac
fsmesh = convObj.fsmesh

# test 1 - initialize grid coordinates
copyGrid2GridInit(fsmesh, 'cart', motionDict, None)
if check: printDiffRigidMotion(t, fsmesh, None, 1)
runTest(fsmesh, 1)

# test 2 - simple translation
motionDict['cart'] = {
    'transl_speed': [1.,0.,0.],
    'axis_pnt': [0.5, 0.5, 0.5],
    'axis_vct': [0.,1.,0.],
    'angular_frq': 0.038
}
evalPosition(fsmesh, 'cart', 0.25, motionDict, None)
if check: printDiffRigidMotion(t, fsmesh, motionDict, 2)
runTest(fsmesh, 2)

# test 3 - simple rotation
motionDict['cart'] = {
    'transl_speed': [0.,0.,0.],
    'axis_pnt': [0.5, 0.5, 0.5],
    'axis_vct': [0.,1.,0.],
    'angular_frq': 0.038
}
evalPosition(fsmesh, 'cart', 0.25, motionDict, None)
if check: printDiffRigidMotion(t, fsmesh, motionDict, 3)
runTest(fsmesh, 3)

# test 4 - rotation + translation
motionDict['cart'] = {
    'transl_speed': [1.,0.,0.],
    'axis_pnt': [0.5, 0.5, 0.5],
    'axis_vct': [0.,1.,0.],
    'angular_frq': 0.038
}
evalPosition(fsmesh, 'cart', 0.25, motionDict, None)
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
evalPosition(fsmesh, 'cart', 0.25, motionDict, None)
runTest(fsmesh, 5)