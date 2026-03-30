# Import modules
import numpy

import Converter.PyTree as C
import Connector.PyTree as X
import Converter.Internal as Internal
import Converter.Mpi as Cmpi
import Generator.PyTree as G
import Transform.PyTree as T
import Geom.PyTree as D

from FSDataManager import (
    FSIntArray, FSStringArray,
    FSUnstructVolumeCellTypes,
    FSDataSpecArray, FSDatasetInfo,
    FSError, FS_AT_CADGroupID
)

import FSOversetBlanking # needed for extractActiveSubMesh and copySolution

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

__all__ = [
    'FSOverset', 'generateBlankingMask', 'extractPyTree',
    'generateDiscParasFromMesh', 'generateBCDictFromMesh',
    'extractActiveSubMesh', 'copySolution'
]

# ---------------------------------------------------------------------------- #
# Classes
# ---------------------------------------------------------------------------- #

class FSOverset:

    def __init__(self, clac, fsmesh, pyTree=None):
        self.clac = clac
        self.fsmesh = fsmesh
        self.pyTree = pyTree
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

def generateBlankingMask(clac, fsmesh, offsets, meshID, meshIDBackground=0, offsetFromBC='BCOverset', dim=3, localDir='./', check=False):
    """Generate a blanking mask from a specified BC"""
    validBCNames = ['BCOverset', 'BCWall']
    if offsetFromBC not in validBCNames:
        raise ValueError("generateBlankingMask: invalid offsetFromBC (%s). Possible values are %s"%(offsetFromBC, validBCNames))

    tb = None

    # Conversion of the curvilinear mesh of the body ('standard' conversion)
    if meshID != meshIDBackground:
        bcDict = generateBCDictFromMesh(fsmesh)
        if fsmesh.HasUnstructDataset('UndeformedCoordinates'): coordsName = 'UndeformedCoordinates'
        else: coordsName = 'Coordinates'
        convObj = FSCGNSConverter(clac=clac, fsmesh=fsmesh, bcDict=bcDict, coordsName=coordsName, datasets=[])
        convObj.convert2CGNS()

        z_body = Internal.getZones(convObj.pyTree)[0]
        z_body[0] += f'.{Cmpi.rank:d}'

        # Extract the BC
        wall = C.extractBCOfType(z_body, offsetFromBC)
        C._deleteFlowSolutions__(wall)
        elts = Internal.getNodesFromType(wall, 'Elements_t')
        for elt in elts:
            if elt[0].startswith('GridElements'):
                Internal._rmNode(wall, elt)
        tb = C.convertArray2Tetra(wall)
        if tb:
            # param_solver = Internal.getNodeFromType(tb, 'UserDefinedData_t')
            # proc_safe = Internal.getNodeFromName(param_solver, 'proc')[1][0][0]
            # tb = T.join(tb)
            # tb = Internal.getZones(tb)[0]
            # param_solver = Internal.newUserDefinedData('param', parent=tb)
            # Internal.newDataArray('.Solver#Param', parent=param_solver, value=proc_safe)
            # Internal.newDataArray('meshID', parent=param_solver, value=meshID)
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
        BB = G.bbox(bodies_offset[meshID])
        xmin = BB[0]; ymin = BB[1]; zmin = BB[2]
        xmax = BB[3]; ymax = BB[4]; zmax = BB[5]
        dmax = max((xmax-xmin), (ymax-ymin), (zmax-zmin))
        ppul = 50./dmax
        if Cmpi.master: print('generateBlankingMask: generating offset (meshID=%d) with ppul=%f and dmax=%f'%(meshID, ppul, dmax))
        bodies_offset[meshID] = D.offsetSurface(bodies_offset[meshID], offset=sign_offset*offsets[meshID-1], pointsPerUnitLength=ppul, algo=0, dim=dim)[0]
        if Cmpi.master and check: C.convertPyTree2File(bodies_offset[meshID], localDir+'wall_offset_%s.plt' %meshID)
        bodies_offset[meshID] = C.convertArray2Tetra(bodies_offset[meshID])
        bodies_offset[meshID] = G.close(bodies_offset[meshID])

    return bodies_offset

def extractPyTree(clac, fsmesh, meshID, meshIDTarget):
    """Extract a pyTree mesh from a fsmesh based on meshID"""
    t = None
    if meshID == meshIDTarget:
        # Conversion of the background mesh ('light' conversion -> only the volume element types)
        convObj = FSCGNSConverter(clac=clac, fsmesh=fsmesh)
        convObj.convert2CGNS(forOverset=True)
        z = Internal.getZones(convObj.pyTree)[0]
        z = C.breakConnectivity(z)
        t = C.newPyTree(['Base', z])
    return t

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