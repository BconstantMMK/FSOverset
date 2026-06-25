# Import modules
import numpy

import Converter.PyTree as C
import Connector.PyTree as X
import Converter.Internal as Internal
import Converter.Mpi as Cmpi
import Generator.PyTree as G
import Transform.PyTree as T
import Geom.PyTree as D
import RigidMotion.PyTree as R
import CPlot.PyTree as CPlot
import CPlot.Decorator as Decorator

from FSDataManager import (
    FSIntArray, FSStringArray, FSFloatArray,
    FSUnstructVolumeCellTypes,
    FSDataSpecArray, FSDatasetInfo,
    FSError, FS_AT_CADGroupID,
    FSMeshEnums, FSDataName
)

import FSOversetBlanking # needed for extractActiveSubMesh and copySolution

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

import math

__all__ = [
    'FSOverset', 'generateBlankingMask', 'extractPyTree',
    'generateDiscParasFromMesh', 'generateBCDictFromMesh',
    'extractActiveSubMesh', 'copySolution'
]

__DEG2RAD__ = math.pi/180.
__RAD2DEG__ = 180./math.pi

# ---------------------------------------------------------------------------- #
# Classes
# ---------------------------------------------------------------------------- #

class FSOverset:

    def __init__(self, clac, fsmesh, meshID, meshIDTarget, pyTree=None):
        self.clac = clac
        self.fsmesh = fsmesh

        self.pyTree = pyTree if pyTree is not None else extractPyTree(clac=clac, fsmesh=fsmesh, meshID=meshID, meshIDTarget=meshIDTarget)
        self.cellNName = 'cellN' # cellN located at the nodes 

        self.fsVolumeCellTypes = FSIntArray(0)
        for cellType in FSUnstructVolumeCellTypes:
            if self.fsmesh.HasCellType(cellType):
                self.fsVolumeCellTypes.Append(cellType)

    def computeBlanking(self, tb, blankingType='center_in', dim=3):
        validBlankingTypes = ['center_in', 'node_in', 'cell_intersect']
        if blankingType not in validBlankingTypes:
            raise ValueError("computeBlanking: invalid blankingType (%s). Possible values are %s"%(blankingType, validBlankingTypes))
        
        if self.pyTree is not None:
            C._deleteEmptyZones(self.pyTree)
            for meshID in tb:
                bodies = [[tb[meshID]]]
                self.pyTree = X.blankCellsTri(self.pyTree, bodies, [], blankingType=blankingType, cellNName=self.cellNName)
             
            # Create an FSDM dataset for cellN obtained in Cassiopee
            cellNList = Internal.getNodesFromName(self.pyTree, self.cellNName)
            cellN = [n_cellN[1] for n_cellN in cellNList]
            np_cellN = numpy.concatenate(cellN)
            Internal._rmNodesFromType(self.pyTree, 'FlowSolution_t')

            if not self.fsmesh.HasUnstructDataset(self.cellNName):
                fsVarNames = FSStringArray(1)
                fsVarNames[0] = self.cellNName
                fsVarSpecs = FSDataSpecArray(1)
                self.fsmesh.InitUnstructDataset(
                    self.cellNName,
                    FSDatasetInfo(fsVarNames, fsVarSpecs, self.fsVolumeCellTypes)
                )

            fs_var = self.fsmesh.GetUnstructDataset(self.cellNName).GetValues()
            numpy.copyto(
                numpy.array(fs_var.Buffer(), copy=False),
                np_cellN[:,None],
                casting='same_kind'
            )

        return None

# ---------------------------------------------------------------------------- #
# FSOverset Functions
# ---------------------------------------------------------------------------- #
#dictOfOffsets : mandatory (can be zero) to specify if a BC defines a blanking mask or not.
def generateBlankingMask(clac, fsmesh, meshID, dictOfOffsets, offsetFromBC='BCOverset', dim=3, localDir='./', check=False):
    """Generate a blanking mask from a specified BC"""
    validBCNames = ['BCOverset', 'BCWall']
    if offsetFromBC not in validBCNames:
        raise ValueError("generateBlankingMask: invalid offsetFromBC (%s). Possible values are %s"%(offsetFromBC, validBCNames))

    tb = None

    # Conversion of the curvilinear mesh of the body ('standard' conversion)
    if meshID in dictOfOffsets:
        bcDict = generateBCDictFromMesh(fsmesh)
        if any(bcDict[bc].startswith(offsetFromBC) for bc in bcDict):
            if fsmesh.HasUnstructDataset('UndeformedCoordinates'): coordsName = 'UndeformedCoordinates'
            else: coordsName = 'Coordinates'
            convObj = FSCGNSConverter(clac=clac, fsmesh=fsmesh, bcDict=bcDict, coordsName=coordsName, datasets=[])
            convObj.convert2CGNS()

            z_body = Internal.getZones(convObj.pyTree)[0]
            z_body[0] += f'.{Cmpi.rank:d}'
            C._deleteFlowSolutions__(z_body)

            # Extract the BC
            wall = C.extractBCOfType(z_body, offsetFromBC)
            del z_body
            if wall != []:        
                elts = Internal.getNodesFromType(wall, 'Elements_t')
                for elt in elts:
                    if elt[0].startswith('GridElements'):
                        Internal._rmNode(wall, elt)
                tb = C.convertArray2Tetra(wall)
                del wall

                tb = T.join(tb)
                Cmpi._setProc(tb, Cmpi.rank)
                param = Internal.getNodeFromName1(tb, '.Solver#Param')
                Internal.newDataArray('meshID', parent=param, value=meshID)

    # Create bodies per meshID
    tb = Cmpi.allgatherZones(tb)
    if Cmpi.master and check: C.convertPyTree2File(tb, localDir+'wall_%d.plt'%meshID)
    bodies = {}
    for zone in Internal.getZones(tb):
        meshID = Internal.getNodeFromName(zone,'meshID')[1][0]
        if meshID not in bodies.keys():
            bodies[meshID] = zone
        else:
           bodies[meshID] = T.join(bodies[meshID],zone)
           bodies[meshID] = G.close(bodies[meshID])

    # Create offset bodies per meshID
    bodies_offset = bodies.copy()
    sign_offset = 1. if offsetFromBC == 'BCWall' else -1.
    for meshID in bodies_offset:
        offsetdist = dictOfOffsets[meshID]
        if offsetdist > 0.:
            BB = G.bbox(bodies_offset[meshID])
            xmin = BB[0]; ymin = BB[1]; zmin = BB[2]
            xmax = BB[3]; ymax = BB[4]; zmax = BB[5]
            dmax = max((xmax-xmin), (ymax-ymin), (zmax-zmin))
            ppul = 50./dmax
            if Cmpi.master: print('generateBlankingMask: generating offset (meshID=%d) with ppul=%f and dmax=%f'%(meshID, ppul, dmax))
            bodies_offset[meshID] = D.offsetSurface(bodies_offset[meshID], offset=sign_offset*dictOfOffsets[meshID], pointsPerUnitLength=ppul, algo=0, dim=dim)[0]
            if Cmpi.master and check: C.convertPyTree2File(bodies_offset[meshID], localDir+'wall_offset_%s.plt' %meshID)
            bodies_offset[meshID] = C.convertArray2Tetra(bodies_offset[meshID])
        else: 
            bodies_offset[meshID] = C.convertArray2Tetra(bodies[meshID])

        bodies_offset[meshID] = G.close(bodies_offset[meshID])

    return bodies_offset

def extractPyTree(clac, fsmesh, meshID, meshIDTarget):
    """Extract a pyTree mesh from a fsmesh based on meshID"""
    t = None
    if meshID == meshIDTarget:
        # Conversion of the blanked mesh ('light' conversion -> only the volume element types)
        convObj = FSCGNSConverter(clac=clac, fsmesh=fsmesh)
        convObj.convert2CGNS(forOverset=True)
        z = Internal.getZones(convObj.pyTree)[0]
        z = C.breakConnectivity(z)
        t = C.newPyTree(['Base', z])
    return t

def display(clac, fsmesh, meshID, variables, it=0, displayDict={}, localDir='./', saveTree=False):
    # get display information
    colormap = displayDict.get('colormap', 24) # default: jet
    isoEdges = displayDict.get('isoEdges', 0.) # line width of isolines
    isoScales = displayDict.get('isoScales', {}) # dict of {varname: [varname, niso, min, max]}
    ppw = displayDict.get('ppw', 1000) # pixels per height
    mpl = displayDict.get('mpl', False) # pixels per height

    # automatically set camera information
    xlim = displayDict['xlim']
    ylim = displayDict['ylim']
    zplane = displayDict['zplane']
    posCam, posEye, dirCam, viewAngle, exportResolution = Decorator.getInfo2DMode(xlim, ylim, zplane, ppw)

    # conversion
    bcDict = generateBCDictFromMesh(fsmesh)
    convObj = FSCGNSConverter(clac=clac, fsmesh=fsmesh, bcDict=bcDict, datasets=['State'])
    convObj.convert2CGNS()
    zone = Internal.getZones(convObj.pyTree)[0]
    Cmpi._setProc(zone, Cmpi.rank)
    zone[0] = '%d_%d'%(meshID, Cmpi.rank)

    listOfMeshID = set(Cmpi.allgather(meshID))
    listOfMeshID = sorted(listOfMeshID)
    listOfZones = []
    for i in listOfMeshID:
        listOfZones.extend([
            'MeshID%d'%i,
            zone if meshID == i else []
        ])
    
    t = C.newPyTree(listOfZones)
    if saveTree: Cmpi.convertPyTree2File(t, localDir+'solution_it%04d.cgns'%it)

    # force (x,y) plane
    T._rotate(t, (0,0,0), (1,0,0), -90.) # from (x,z) to (x,y)

    # temporary patch for intra-grid match connection
    t = Cmpi.allgatherTree(t)
    if Cmpi.master:
        listOfZones = []
        for b in Internal.getBases(t): 
            zone = T.join(Internal.getZones(b))
            listOfZones.extend([b[0],zone])
        t = C.newPyTree(listOfZones)
    else:
        t = []

    # display
    for v in variables:
        filename = localDir+'%s_it%04d.png'%(v, it)
        export = CPlot.decorator if mpl else filename

        if v not in isoScales:
            vmin = Cmpi.getMinValue(t, 'centers:%s'%v)
            vmax = Cmpi.getMaxValue(t, 'centers:%s'%v)
            isoScales[v] = [v, 25, vmin, vmax] # default CPlot values

        CPlot.display(t, mode='Scalar', scalarField=v,
            dim=2, export=export, isoScales=isoScales[v], isoEdges=isoEdges,
            offscreen=7, bgColor=0, colormap=colormap,
            viewAngle=viewAngle,
            posCam=posCam, posEye=posEye, dirCam=dirCam,
            exportResolution=exportResolution)
        
        if mpl and Cmpi.master:
            fig, ax = Decorator.createSubPlot(box=True, figsize=(7,6), dpi=100, xlim=xlim, ylim=ylim)
            cbar = Decorator.createColorBar(fig, ax, title=v, discrete=True, nticks=5, labelFormat='%.2f', size='3%')
            Decorator.savefig(filename, pad=0.1, dpi=200)
    
    return None

# ---------------------------------------------------------------------------- #
# FSOversetBlanking Functions
# ---------------------------------------------------------------------------- #

def extractActiveSubMesh(fsdatamanager, MeshKeyOrig, meshKeyActive):
    """Extract the active mesh (blanked mesh) from the original mesh (non-blanked mesh) based on cellN dataset"""
    # cellN = 0: blanked cells
    # cellN = 1: non-blanked cells
    dataManagerOps = (('BlankMesh', {'MeshKeyOrig'     : MeshKeyOrig,
                                     'MeshKeyActive'   : meshKeyActive,
                                     'DatasetNameCellNature': 'cellN',
                                     'QuantityNameCellNature': 'cellN',
                                     'CellNatureActive': 1,
                                     'AddFacesMarker': 56,
                                     'AddFacesBC': 'BCOverset',
                                    }),)
    fsdatamanager.DoOps(dataManagerOps) or FSError.PrintAndExit()
    return None

def copySolution(fsdatamanager, MeshKeyOrig, meshKeyActive):
    """Copy the flow solution from the active mesh (blanked mesh) to the original mesh (non-blanked mesh)"""
    # cellN = 0: blanked cells
    # cellN = 1: non-blanked cells
    dataManagerOps = (('CopyDataOfBlankedMesh', {'MeshKeyOrig'    : MeshKeyOrig,
                                                 'MeshKeyActive'  : meshKeyActive,
                                                 'AttributeNameDataAvailable': 'DataPresent',
                                                 'InitBlankedCellData': 2,
                                          }),)
    fsdatamanager.DoOps(dataManagerOps) or FSError.PrintAndExit()
    return None

# ---------------------------------------------------------------------------- #
# Helper Functions
# ---------------------------------------------------------------------------- #

def getBoundaryTreatmentsFromMesh__(fsmesh):
    markers = FSIntArray()
    fsmesh.GatherCellAttributeValues(FS_AT_CADGroupID, markers) or FSError.PrintAndExit()

    treatments = {}
    for marker in markers: # get BCType and associated boundary markers
        treatmentType = str(fsmesh.GetCellAttributeValueName(FS_AT_CADGroupID, marker))
        treatmentType = ''.join([i for i in treatmentType if not i.isdigit()]) # remove integer(s) from name
        if treatmentType in treatments: treatments[treatmentType].append(marker)
        else: treatments[treatmentType] = [marker]
    
    return treatments

def generateDiscParasFromMesh(fsmesh, discParaDict=None):
    """Extract the boundary marker names from the fsmesh and generates the matching discParas"""
    treatments = getBoundaryTreatmentsFromMesh__(fsmesh)
    paraDict = discParaDict.copy() if discParaDict is not None else {}
    paraDict['boundary treatments'] = [
        {'treatment type': key, 'boundary markers': value} for key, value in treatments.items()
    ]
    return paraDict

def generateBCDictFromMesh(fsmesh):
    """Extract the boundary marker names from the fsmesh and generates the matching bcdict"""
    treatments = getBoundaryTreatmentsFromMesh__(fsmesh)
    bcDict = {}
    for treatmentType, markers in treatments.items():
        for marker in markers: bcDict[marker] = treatmentType
    return bcDict

def getWallBoundaryMarkers(fsmesh):
    treatments = getBoundaryTreatmentsFromMesh__(fsmesh)
    wallMarkers = []
    for key, value in treatments.items():
        if 'Wall' in key: wallMarkers.extend(value)
    return wallMarkers

def initGridVelocity(fsmesh):
    if not fsmesh.HasUnstructDataset('GridVelocity'):
        nNodes = fsmesh.GetNCells(FSMeshEnums.CT_Node)
        gridVelNames = FSStringArray(3)
        gridVelNames[0] = FSDataName.GridVelocity().X()
        gridVelNames[1] = FSDataName.GridVelocity().Y()
        gridVelNames[2] = FSDataName.GridVelocity().Z()
        gridVelSpecs = FSDataSpecArray(3)
        gridVelSpecs[0].Velocity()
        gridVelSpecs[1].Velocity()
        gridVelSpecs[2].Velocity()
        gridVels = FSFloatArray(nNodes, 3)
        gridVels.Fill(0.0)
        fsmesh.InitUnstructDataset('GridVelocity', FSDatasetInfo(gridVelNames, gridVelSpecs, FSMeshEnums.CT_Node), gridVels)

    return None

def copyGrid2GridInit(fsmesh=None, mask=None):
    if fsmesh is not None:
        if not fsmesh.HasUnstructDataset('UndeformedCoordinates'):
            coordsDataset = fsmesh.GetUnstructDataset('Coordinates')
            coords = coordsDataset.GetValues()
            undeformedNames = FSStringArray(3)
            undeformedNames[0] = FSDataName.Coordinates().X()
            undeformedNames[1] = FSDataName.Coordinates().Y()
            undeformedNames[2] = FSDataName.Coordinates().Z()
            undeformedSpecs = FSDataSpecArray(3)
            undeformedSpecs[0].Length()
            undeformedSpecs[1].Length()
            undeformedSpecs[2].Length()
            undeformed = coords
            fsmesh.InitUnstructDataset('UndeformedCoordinates', FSDatasetInfo(undeformedNames, undeformedSpecs, FSMeshEnums.CT_Node), undeformed)
    
    if mask is not None:
        for meshID in mask:
            z = mask[meshID]
            R._copyGrid2GridInit(z, mode=1)

    return None

def copyGridInit2Grid(fsmesh):
    refCoords = fsmesh.GetUnstructDataset('UndeformedCoordinates').GetValues()
    gridCoords = fsmesh.GetUnstructDataset('Coordinates').GetValues()
    numpy.copyto(
            numpy.array(gridCoords.Buffer(), copy=False),
            numpy.array(refCoords.Buffer(), copy=False),
            casting='same_kind'
    )

    return None

def evalPosition(fsmesh=None, mask=None, time=0, motionDict=None):
    tx, ty, tz = motionDict['transl_speed']
    cx, cy, cz = motionDict['axis_pnt']
    kx, ky, kz = motionDict['axis_vct']
    omega = motionDict['angular_frq']

    if 'ampl_angle' in motionDict: # oscillation
        alphaMean = motionDict['mean_angle']
        alphaAmpl = motionDict['ampl_angle']
        alpha = alphaMean + alphaAmpl * math.sin(omega * time)
    else: # rotation
        alpha = omega * time * __RAD2DEG__
    
    cosalpha = math.cos(alpha * __DEG2RAD__)
    sinalpha = math.sin(alpha * __DEG2RAD__)

    if mask is not None:
        for meshID in mask:
            z = mask[meshID]
            R._copyGridInit2Grid(z)
            T._rotate(z, (cx,cy,cz), (kx,ky,kz), alpha, vectors=[])
            T._translate(z, (tx*time, ty*time, tz*time))
            
    if fsmesh is not None:
        nNodes = fsmesh.GetNCells(FSMeshEnums.CT_Node)
        copyGridInit2Grid(fsmesh)
        gridCoords = fsmesh.GetUnstructDataset('Coordinates').GetValues()

        for node in range(nNodes):
            x = gridCoords[3 * node]
            y = gridCoords[3 * node + 1]
            z = gridCoords[3 * node + 2]

            # position vector
            cmx = x - cx
            cmy = y - cy
            cmz = z - cz

            # k x CM
            kcmx = ky * cmz - kz * cmy
            kcmy = kz * cmx - kx * cmz
            kcmz = kx * cmy - ky * cmx

            # k . CM
            kcm = kx * cmx + ky * cmy + kz * cmz

            # rotation (Rodrigues' rotation formula) + translation
            x = (cx + cosalpha * cmx + (1 - cosalpha) * kcm * kx + sinalpha * kcmx) + tx
            y = (cy + cosalpha * cmy + (1 - cosalpha) * kcm * ky + sinalpha * kcmy) + ty
            z = (cz + cosalpha * cmz + (1 - cosalpha) * kcm * kz + sinalpha * kcmz) + tz

            gridCoords[3 * node] = x
            gridCoords[3 * node + 1] = y
            gridCoords[3 * node + 2] = z

    return None

def evalGridSpeed(fsmesh=None, time=0, motionDict=None):
    tx, ty, tz = motionDict['transl_speed']
    cx, cy, cz = motionDict['axis_pnt']
    kx, ky, kz = motionDict['axis_vct']
    omega = motionDict['angular_frq']

    if 'ampl_angle' in motionDict: # oscillation
        alphaAmpl = motionDict['ampl_angle']
        alphaDot = omega * alphaAmpl * math.cos(omega * time) # derivative of alpha w.r.t to time
        alphaDot *= __DEG2RAD__ # radians per sec.
    else: # rotation
        alphaDot = omega

    if fsmesh is not None:
        nNodes = fsmesh.GetNCells(FSMeshEnums.CT_Node)
        gridCoords = fsmesh.GetUnstructDataset('Coordinates').GetValues() # grid has already been moved
        gridVels = fsmesh.GetUnstructDataset('GridVelocity').GetValues()
        
        for node in range(nNodes):
            x = gridCoords[3 * node]
            y = gridCoords[3 * node + 1]
            z = gridCoords[3 * node + 2]

            # position vector
            cmx = x - cx
            cmy = y - cy
            cmz = z - cz

            # k x CM
            kcmx = ky * cmz - kz * cmy
            kcmy = kz * cmx - kx * cmz
            kcmz = kx * cmy - ky * cmx

            # grid speed
            vx = tx + alphaDot * kcmx
            vy = ty + alphaDot * kcmy
            vz = tz + alphaDot * kcmz

            gridVels[3 * node] = vx
            gridVels[3 * node + 1] = vy
            gridVels[3 * node + 2] = vz

    return None

# ---------------------------------------------------------------------------- #
# Deprecated Functions
# ---------------------------------------------------------------------------- #

def surfaceBackgroundMesh(clac, fsmesh, paraDict, wallBoundaryMarkers, offsets, meshID):
    Cmpi.barrier()
    tb2 = None
    # Conversion of the curvilinear mesh of the body ("standard" conversion)
    if meshID == 0:
        tmp_bcDict = {}
        for dictio in paraDict["boundary treatments"]:
            if dictio["treatment type"] in tmp_bcDict:
                for value in dictio["boundary markers"]:
                    tmp_bcDict[dictio["treatment type"]].append(value)
            else:
                tmp_bcDict[dictio["treatment type"]] = dictio["boundary markers"]
        bcDict = {}
        for bcname, markers in tmp_bcDict.items():
            for marker in markers:
                bcDict[marker] = bcname

        convObj = FSCGNSConverter(clac=clac, fsmesh=fsmesh, bcDict=bcDict)
        convObj.convert2CGNS()
        z_body = Internal.getZones(convObj.pyTree)[0]
        z_body[0] += "."+str(Cmpi.rank)

        # Extract only the wall for the blanking (surface mesh)
        if Cmpi.master: print(Cmpi.rank, "zonevalue", Internal.getValue(z_body))
        ER = Internal.getNodeFromName(z_body, "ElementRange")[1]

        if (ER[1] - ER[0] + 1) > 0:
            extFaces = z_body
            Internal._rmNodesFromType(extFaces,"ZoneBC_t")
            elts_extFaces = Internal.getNodesFromType(extFaces,"Elements_t")
            for elt in elts_extFaces:
                if elt[0].startswith("GridElements"):
                    Internal._rmNode(extFaces,elt)
            zones = []
            elts_extFaces = Internal.getNodesFromType(extFaces,"Elements_t")

            zones = []
            for elt_eF in elts_extFaces:
                extFaces_select = C.selectConnectivity(extFaces,elt_eF[0])
                extFaces_select = C.convertArray2Tetra(extFaces_select)
                zones.append(extFaces_select)

            tb2 = T.join(zones)

    tb2 = Cmpi.allgatherZones(tb2)
    bodies = {}
    for zone in Internal.getZones(tb2):
        #meshid = Internal.getNodeFromName(zone,"meshID")[1][0]
        meshid = meshID
        if meshid not in bodies:
            bodies[meshid] = zone
        else:
            bodies[meshid] = T.join(bodies[meshid],zone)
            bodies[meshid] = G.close(bodies[meshid])
    bodies_offset = bodies.copy()
    for meshid in bodies_offset:
        BB = G.bbox(bodies_offset[meshid])
        xmin, ymin, zmin = BB[:3]
        xmax, ymax, zmax = BB[3:]
        dmax = max((xmax-xmin), (ymax-ymin), (zmax-zmin))
        ppul = 100./dmax
        if Cmpi.master: print("Points per unit lenght=", ppul)
        bodies_offset[meshid] = D.offsetSurface(bodies_offset[meshid], offset=-offsets[meshid-1], pointsPerUnitLength=ppul, algo=0, dim=3)[0]
        C.convertPyTree2File(bodies_offset[meshid], "wall_OVERSET_%s.plt" %meshid)
        bodies_offset[meshid] = C.convertArray2Tetra(bodies_offset[meshid])
        bodies_offset[meshid] = G.close(bodies_offset[meshid])
    #offset_tb2 = C.newPyTree(["bodies",list(bodies.values())])
    return bodies, bodies_offset