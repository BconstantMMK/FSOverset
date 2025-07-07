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
    FSDataSpecArray, FSDatasetInfo
)

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

__all__ = [
    'FSOverset',
    'holeMesh', 'surfaceBackgroundMesh', 'backgroundMesh', 'childMesh'
]


# ---------------------------------------------------------------------------- #
# Classes
# ---------------------------------------------------------------------------- #

class FSOverset:

    validBlankingTypes = ["center_in", "node_in", "cell_intersect"]

    def __init__(self, clac, fsmesh, pyTree=None):
        self.clac = clac
        self.fsmesh = fsmesh
        self.t_bg = pyTree

        self.fsVolumeCellTypes = FSIntArray(0)
        for cellType in FSUnstructVolumeCellTypes:
            if self.fsmesh.HasCellType(cellType):
                self.fsVolumeCellTypes.Append(cellType)

    def computeBlanking(self, tb2, blankingType="center_in"):
        if blankingType not in FSOverset.validBlankingTypes:
            raise ValueError(
                f"Invalid blankingType: {blankingType}. "
                f"Must be one of {FSOverset.validBlankingTypes}."
            )
        if self.t_bg is not None:
            C._deleteEmptyZones(self.t_bg)
            for meshid in tb2:
                bodies = [[tb2[meshid]]]
                self.t_bg = X.blankCellsTri(self.t_bg, bodies, [], blankingType=blankingType)

            # Create an FSDM dataset for cellN obtained in Cassiopee
            cellNList = Internal.getNodesFromName(self.t_bg, "cellN")
            cellN = [n_cellN[1] for n_cellN in cellNList]
            np_cellN = numpy.concatenate(cellN)
            Internal._rmNodesFromType(self.t_bg, "FlowSolution_t")

            varName = "cellN"
            if not self.fsmesh.HasUnstructDataset(varName):
                fsVarNames = FSStringArray(1)
                fsVarNames[0] = varName
                fsVarSpecs = FSDataSpecArray(1)
                self.fsmesh.InitUnstructDataset(
                    varName,
                    FSDatasetInfo(fsVarNames, fsVarSpecs, self.fsVolumeCellTypes)
                )

            fs_var = self.fsmesh.GetUnstructDataset(varName).GetValues()
            numpy.copyto(
                numpy.array(fs_var.Buffer(), copy=False),
                np_cellN[:,None],
                casting='same_kind'
            )

        return None


# ---------------------------------------------------------------------------- #
# Functions
# ---------------------------------------------------------------------------- #

def holeMesh(clac, fsmesh, paraDict, offsets, meshID, offsetFromBC="BCOverset"):
    tb2 = None
    # Conversion of the curvilinear mesh of the body ("standard" conversion)
    if meshID > 0:
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
        if fsmesh.HasUnstructDataset("UndeformedCoordinates"):
            coordsName = "UndeformedCoordinates"
        else:
            coordsName = "Coordinates"
        convObj = FSCGNSConverter(clac=clac, fsmesh=fsmesh, bcDict=bcDict, coordsName=coordsName)
        convObj.convert2CGNS()

        z_body = Internal.getZones(convObj.pyTree)[0]
        z_body[0] += f".{Cmpi.rank:d}"

        # Extract only the wall for the blanking (surface mesh)
        wall = C.extractBCOfType(z_body,offsetFromBC)
        C._deleteFlowSolutions__(wall)
        elts = Internal.getNodesFromType(wall, "Elements_t")
        for elt in elts:
            if elt[0].startswith("GridElements"):
                Internal._rmNode(wall, elt)
        tb2 = C.convertArray2Tetra(wall)
        if len(tb2) != 0:
            param_solver = Internal.getNodeFromType(tb2, "UserDefinedData_t")
            proc_safe = Internal.getNodeFromName(param_solver, "proc")[1][0][0]
            tb2 = T.join(tb2)
            tb2 = Internal.getZones(tb2)[0]
            param_solver = Internal.newUserDefinedData("param", parent=tb2)
            Internal.newDataArray(".Solver#Param", parent=param_solver, value=proc_safe)
            Internal.newDataArray("meshID", parent=param_solver, value=meshID)

    tb2 = Cmpi.allgatherZones(tb2)
    if Cmpi.master: C.convertPyTree2File(tb2, "wall.plt")
    bodies = {}
    for zone in Internal.getZones(tb2):
        meshid = Internal.getNodeFromName(zone,"meshID")[1][0]
        if meshid not in bodies.keys():
            bodies[meshid] = zone
        else:
            bodies[meshid] = T.join(bodies[meshid],zone)
            bodies[meshid] = G.close(bodies[meshid])
    bodies_offset = bodies.copy()
    sign_offset = 1. if offsetFromBC=="BCWall" else -1.
    for meshid in bodies_offset:
        BB = G.bbox(bodies_offset[meshid])
        xmin = BB[0]; ymin = BB[1]; zmin = BB[2]
        xmax = BB[3]; ymax = BB[4]; zmax = BB[5]
        dmax = max((xmax-xmin), (ymax-ymin), (zmax-zmin))
        ppul = 100./dmax
        if Cmpi.master: print("Points per unit lenght=",ppul)
        bodies_offset[meshid] = D.offsetSurface(bodies_offset[meshid], offset=sign_offset*offsets[meshid-1], pointsPerUnitLength=ppul, algo=0, dim=3)[0]
        if Cmpi.master: C.convertPyTree2File(bodies_offset[meshid], "wall_offset_%s.plt" %meshid)
        bodies_offset[meshid] = C.convertArray2Tetra(bodies_offset[meshid])
        bodies_offset[meshid] = G.close(bodies_offset[meshid])

    #offset_tb2 = C.newPyTree(["bodies",list(bodies.values())])
    return bodies_offset

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

def backgroundMesh(clac, fsmesh, meshID):
    t_bg = None
    if meshID == 0:
        # Conversion of the background mesh ("light" conversion -> only the volume element types)
        convObj = FSCGNSConverter(clac=clac, fsmesh=fsmesh)
        convObj.convert2CGNS(forOverset=True)
        z_bg = Internal.getZones(convObj.pyTree)[0]
        z_bg = C.breakConnectivity(z_bg)
        #z_bg = C.convertArray2NGon(z_bg)
        #z_bg = T.join(z_bg)
        #z_bg[0] = z_bg[0]+"."+str(Cmpi.rank)
        t_bg = C.newPyTree(['Base', z_bg])
    return t_bg

def childMesh(clac, fsmesh, meshID, meshIDChildMeshToBlank):
    t_bg = None
    if meshID == meshIDChildMeshToBlank:
        # Conversion of the background mesh ("light" conversion -> only the volume element types)
        convObj = FSCGNSConverter(clac=clac, fsmesh=fsmesh)
        convObj.convert2CGNS(forOverset=True)
        z_bg = Internal.getZones(convObj.pyTree)[0]
        z_bg = C.breakConnectivity(z_bg)
        #z_bg = C.convertArray2NGon(z_bg)
        #z_bg = T.join(z_bg)
        #z_bg[0] = z_bg[0]+"."+str(Cmpi.rank)
        t_bg = C.newPyTree(['Base', z_bg])
    return t_bg
