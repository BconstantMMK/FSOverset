import sys
import numpy

import Converter.PyTree as C
import Post.PyTree as P
import Geom.PyTree as D
import Transform.PyTree as T
import Generator.PyTree as G
import Converter.Internal as Internal

from FSDataManager import FSUnstructVolumeCellTypes, FSUnstructSurfaceCellTypes, FSMeshEnums, FS_AT_CADGroupID
from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter, _addBC2ZoneLoc

meshes_names = ["cyl_curvi", "cyl_curvi_bigger"]
external_radii = [0.7, 1]

x_center = 1.8
y_center = 0.0
dz = 0.01
zMin = 0
zMax = dz
size = 0.04
L = 1
N = 200
N_cylinder = int(L/size*2)

for mesh_name,external_radius in zip(meshes_names,external_radii):

    ff = D.circle((x_center,y_center,0), external_radius, N=N)
    ff = C.convertArray2Tetra(ff)
    ff = G.close(ff)

    cylinder = D.circle((x_center,y_center,0), 0.1, N=N_cylinder)
    cylinder = C.convertArray2Tetra(cylinder)
    cylinder = T.reorder(cylinder, (-1,))
    cylinder = G.close(cylinder)

    borders = T.join(cylinder, ff)

    m2 = G.T3mesher2D(borders, triangulateOnly=0, grading=1.05, metricInterpType=0)

    T._addkplane(m2)
    T._contract(m2, (0,0,0), (1,0,0), (0,1,0), dz)

    zones = Internal.getZones(m2)
    tol = 1e-8
    extFaces = P.exteriorFaces(zones[0])
    extFaces = T.breakElements(extFaces)
    Internal.printTree(extFaces)

    list_bcs = ["BCOverset","BCSymmetryPlane","BCWallInviscid"]
    for zone_extFaces in Internal.getZones(extFaces):
        print("Creating boundary condition.. ")
        GE = Internal.getNodeFromName(zone_extFaces, 'GridElements')
        EC = Internal.getNodeFromName(zone_extFaces, 'ElementConnectivity')[1]
        ER = Internal.getNodeFromName(zone_extFaces, 'ElementRange')[1]
        xCoord = Internal.getNodeFromName(zone_extFaces, 'CoordinateX')[1]
        yCoord = Internal.getNodeFromName(zone_extFaces, 'CoordinateY')[1]
        zCoord = Internal.getNodeFromName(zone_extFaces, 'CoordinateZ')[1]

        nb_cell_boundary = zone_extFaces[1][0][1]
        etype = GE[1][0]
        if etype == 5: n_vertices = 3
        elif etype == 7: n_vertices = 4

        EC_reshape = EC.reshape((nb_cell_boundary,n_vertices))
        dictionary_bcs_idx = {k:[] for k in list_bcs}

        for idx in range(nb_cell_boundary):
            CODABCType = None
            if numpy.all(zCoord[EC_reshape[idx]-1]-tol<zMin) or numpy.all(zCoord[EC_reshape[idx]-1]+tol>zMax):
                CODABCType='BCSymmetryPlane'
            elif (numpy.all(((xCoord[EC_reshape[idx]-1]-x_center)**2 + (yCoord[EC_reshape[idx]-1]-y_center)**2)**0.5 + tol>external_radius)):
                CODABCType='BCOverset'
            else:
                CODABCType="BCWallInviscid"
            dictionary_bcs_idx[CODABCType].append(idx)

        for CODABCType in list_bcs:
            print(CODABCType)
            if len(dictionary_bcs_idx[CODABCType])>0:
                zf = T.subzone(zone_extFaces,dictionary_bcs_idx[CODABCType], type='elements')
                _addBC2ZoneLoc(zones[0], CODABCType, CODABCType, zf)

    convObj = FSCGNSConverter(pyTree=m2, flipYZAxes=True)
    convObj.convert2FSDM()
    volumeCellTypes = tuple(FSMeshEnums.CellTypeToString(x) for x in FSUnstructVolumeCellTypes)
    convObj.fsmesh.ExportMeshTECPLOT(Filename=mesh_name+"_vol.plt", PrefixDatasetName=True,  ExportCellTypes=volumeCellTypes) or FSError.PrintAndExit()

    surfaceCellTypes = tuple(FSMeshEnums.CellTypeToString(x) for x in FSUnstructSurfaceCellTypes)
    convObj.fsmesh.ExportMeshTECPLOT(Filename=mesh_name+"_surf.plt", PrefixDatasetName=True, ZonePerCellAttributeValue=True,CellAttribute=FS_AT_CADGroupID, UseCellAttributeValueName=True,ExportCellTypes=surfaceCellTypes) or FSError.PrintAndExit()

    convObj.export(filename=mesh_name+".h5")
