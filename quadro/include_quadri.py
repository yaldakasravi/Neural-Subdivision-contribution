import os
import sys

cwd = os.getcwd()
sys.path.append(cwd + '/pythongptoolbox/')
sys.path.append(cwd + '/torchgptoolbox_nosparse/')

# torch
import torch
import torchgptoolbox_nosparse as tgp

# pygptoolbox
from readOBJ import readOBJ
from writeOBJ import writeOBJ
from findIdx import findIdx
from midPointUpsampling import midPointUpsampling

# python standard 
import numpy as np
import scipy
import scipy.sparse
import glob
import time
import sys
import json
import pickle as pickle


#NOTE YALDA rewrote the whole helper fucnitons to no help with quad meshes
# NOTE built on TOP of include from orignal code
#NOTE supports building , reading , helper functions for  quadrimeshes


class Mesh: 
    # store information of a mesh
    def __init__(self, V, F, hfList):
        """
        Inputs:
            V: nV-3 vertex list
            F: nF-4 face list
            hfList: nHF-4 ordered vertex index of all half flaps 
            
            Notes: 
            each half flap order looks like (see paper for the color scheme)
            [v_blue, v_red, v_purple, v_yellow]
        """
        self.V = V
        self.F = F
        self.hfList = hfList



def processTrainShapes(folder):
    """
    Process training shapes given a folder, including computing the half flap list and reading vertex/face lists.
    This version is updated for quad meshes.
    """
    subFolders = [x for x in os.listdir(folder) if not x.startswith('.')]
    nSubd = len(subFolders) - 1

    objPaths = glob.glob(folder + subFolders[0] + "/*.obj")
    nObjs = len(objPaths)

    paths = [folder + 'subd' + str(ii) + '/' for ii in range(nSubd + 1)]
    objFiles = [str(ii + 1).zfill(3) for ii in range(nObjs)]

    meshes = [None] * nObjs
    for ii in range(nObjs):
        print('Processing mesh %d / %d' % (ii + 1, nObjs))
        meshes_i = [None] * (nSubd + 1)
        for jj in range(nSubd + 1):
            V, F = tgp.readOBJ(paths[jj] + objFiles[ii] + '.obj')
            print(f"Vertices: {V.shape}, Faces: {F.shape}")
            # Adjust `computeFlapList` for quad meshes
            _, hfList = computeFlapList(V, F, 4)  # Use 4 for quads
            print(hfList)
            print(V.shape)
            print(F.shape)
            meshes_i[jj] = Mesh(V, F, hfList[0])
        meshes[ii] = list(meshes_i)
    print('Num Subd: %d, Num Meshes: %d' % (nSubd, nObjs))
    return meshes

class TrainMeshes:
    """
    Store information of many training meshes (updated for quad meshes).
    """
    def __init__(self, folders):
        nShape = len(folders)
        self.meshes = []
        for fIdx in range(nShape):
            meshes = processTrainShapes(folders[fIdx])
            self.meshes.extend(meshes)
        self.nM = len(self.meshes)  # Number of meshes
        self.nS = len(self.meshes[0])  # Number of subdivision levels

        self.hfList = None
        self.poolMats = None
        self.dofs = None
        self.LCs = None

    def getInputData(self, mIdx):
        """
        get input data for the network
        Inputs: 
            mIdx: mesh index
        """
        input = torch.cat((
            self.meshes[mIdx][0].V, # vertex positions
            self.LCs[mIdx]), # vector of differential coordinates
            dim=1)
        return input 

    def computeLaplaceCoordinatesQuad(self,V, faces):
        """
        Compute Laplace coordinates for a quadmesh.
        
        Inputs:
            V: (N, 3) tensor of vertex positions
            faces: (F, 4) tensor of quad faces (vertex indices)
        
        Outputs:
            laplace_coords: (N, 3) tensor of Laplace coordinates
        """
        # Step 1: Generate directed edges
        edges = []
        for f in faces:
            edges.extend([
                [f[0], f[1]],  # Edge v0 -> v1
                [f[1], f[2]],  # Edge v1 -> v2
                [f[2], f[3]],  # Edge v2 -> v3
                [f[3], f[0]],  # Edge v3 -> v0
            ])
        edges = torch.tensor(edges, dtype=torch.long)
    
        # Step 2: Compute edge vectors
        edge_vectors = V[edges[:, 1]] - V[edges[:, 0]]  # Edge vectors v_j - v_i
    
        # Step 3: Accumulate edge vectors at each vertex
        # Create sparse pooling matrix to map edges to vertices
        num_vertices = V.shape[0]
        num_edges = edges.shape[0]
        row_indices = edges[:, 0]  # Start vertices (v_i)
        col_indices = torch.arange(num_edges)  # Edge indices
        values = torch.ones(num_edges, dtype=torch.float32)  # All edges contribute equally
        pooling_matrix = torch.sparse_coo_tensor(
            indices=torch.stack([row_indices, col_indices]),
            values=values,
            size=(num_vertices, num_edges),
        )
    
        # Pool edge vectors
        pooled_vectors = torch.sparse.mm(pooling_matrix, edge_vectors)
    
        # Step 4: Normalize by degree
        degree = pooling_matrix.sum(dim=1).to_dense()  # Degrees of each vertex
        laplace_coords = pooled_vectors / degree.unsqueeze(1)
    
        return laplace_coords
    #def getHalfFlap(self): 
    #    """ 
    #    Create a list of half-flap information for quad-meshes. 
    #    HF[meshIdx][subdIdx] = [v_blue, v_red, v_purple, v_yellow]
    #    Each half-flap consists of 4 vertices from adjacent faces.

    #    Returns:
    #        HF: A list of half-flap lists for each mesh at each subdivision level.
    #    """ 
    #    HF = [None] * self.nM  # Initialize list to store half-flaps for each mesh
    #    
    #    for ii in range(self.nM):
    #        fifj = [None] * (self.nS - 1)  # Temporary list to store half-flaps for each subdivision level
    #        
    #        for jj in range(self.nS - 1):
    #            # Extract half-flap indices: Assuming hfList contains 4 indices for each face.
    #            # For a quad-mesh, each face has four vertices: [v1, v2, v3, v4].
    #            #idx = self.meshes[ii][jj].hfList[:, [0, 1, 2, 3 ]]  # Adjust for quad-mesh indexing
    #            idx = self.meshes[ii][jj].hfList[:, [0, 1, 2, 3 ]]  # Adjust for quad-mesh indexing
    #            
    #            # Reshape half-flap list into the expected 4-vertex structure
    #            fifj[jj] = idx.reshape(-1, 4)  # Flatten into a 2D array where each row represents a half-flap
    #            print(idx)
    #            input("printing idx ")


    #        # Store the result for the current mesh
    #        HF[ii] = list(fifj)

    #        
    #    print(HF[0][0].shape) 
    #    return HF

    #def getHalfFlap(self):
    #    """
    #    Create a list of half flap information for quad meshes.
    #    """
    #    HF = [None] * self.nM
    #    for ii in range(self.nM):
    #        fifj = [None] * (self.nS - 1)
    #        for jj in range(self.nS - 1):
    #            F = self.meshes[ii][jj].F  # Face indices
    #            hfList = []
    #            for face in F:
    #                for i in range(4):  # Loop over 4 edges per quad
    #                    v0 = face[i]
    #                    v1 = face[(i + 1) % 4]
    #                    hfList.append([v0, v1])
    #            hfList = torch.tensor(hfList, dtype=torch.long)
    #            fifj[jj] = hfList
    #        HF[ii] = list(fifj)
    #    return HF

    def getHalfFlap(self):
        """
        Create a list of half flap information for quad meshes.
        """
        HF = [None] * self.nM
        for ii in range(self.nM):
            fifj = [None] * (self.nS - 1)
            for jj in range(self.nS - 1):
                idx = self.meshes[ii][jj].hfList[:, [0, 1, 2, 3]]  # its fine
                fifj[jj] = idx.reshape(-1, 4)
            HF[ii] = list(fifj)
        return HF

    def getFlapPool(self, HF):
        """
        Get the matrix for vertex one-ring average pooling for quad meshes.
        """
        nM = len(HF)
        nS = len(HF[0])

        poolFlap = [None] * nM
        dof = [None] * nM

        for ii in range(nM):
            poolFlap_ij = [None] * nS
            dof_ij = [None] * nS
            #list initalized

            for jj in range(nS):
                #getting the halfflaps we made
                #NOTE why is it going to just ns == 1 and not to ns == 2
                print(f" state {jj}")
                hfIdx = HF[ii][jj]
                # getting max vertice from the half flaps
                nV = hfIdx[:, 0].max() + 1

                rIdx = hfIdx[:, 0]
                cIdx = torch.arange(hfIdx.size(0))
                I = torch.cat([rIdx, cIdx], 0).reshape(2, -1)
                val = torch.ones(hfIdx.size(0))

                print(f" I {I.size()}")
                print(f" val {val.size()}")

                poolMat = torch.sparse.FloatTensor(I, val, torch.Size([nV, hfIdx.size(0)]))
                rowSum = torch.sparse.sum(poolMat, dim=1).to_dense()

                print(f" poolMat {poolMat.size()}")
                print(f" rowSum {rowSum.size()}")

                if jj > 0:
                    input()

                poolFlap_ij[jj] = poolMat
                dof_ij[jj] = rowSum
            poolFlap[ii] = list(poolFlap_ij)
            dof[ii] = list(dof_ij)
        return poolFlap, dof

    def getLaplaceCoordinate(self, hfList, poolMats, dofs):
        """
        get the vectors of the differential coordinates (see Fig.18)
        Inputs:
            hfList: half flap list (see self.getHalfFlap)
            poolMats: vertex one-ring pooling matrix (see self.getFlapPool)
            dofs: degrees of freedom per vertex (see self.getFlapPool)
        """
        LC = [None] * self.nM
        for mIdx in range(self.nM):
            V = self.meshes[mIdx][0].V
            #faces = self.meshes[mIdx][0].F
            HF = hfList[mIdx][0]

            poolMat = poolMats[mIdx][0]
            dof = dofs[mIdx][0]

            dV_he = V[HF[:, 0], :] - V[HF[:, 1], :]
            dV_v = torch.spmm(poolMat, dV_he)
            dV_v /= dof.unsqueeze(1)
            LC[mIdx] = dV_v

            #TODO remove
            #tmp_LC = self.computeLaplaceCoordinatesQuad(V, faces)
            #print(tmp_LC.shape)
            #input()
            #LC[mIdx] = tmp_LC
        return LC

    def computeParameters(self):

        """
        pre-compute parameters required for network training. It includes:
        hfList: list of half flaps
        poolMats: vertex one-ring pooling 
        LCs: vector of differential coordinates
        """

        #print(" we are here ")
        #input()
        self.hfList = self.getHalfFlap()
        self.poolMats, self.dofs = self.getFlapPool(self.hfList)
        self.LCs = self.getLaplaceCoordinate(self.hfList, self.poolMats, self.dofs)
        #self.LCs = self.

    def toDevice(self, device):
        """
        move information to CPU/GPU
        """
        for ii in range(self.nM):
            for jj in range(self.nS):
                self.meshes[ii][jj].V = self.meshes[ii][jj].V.to(device)
                self.meshes[ii][jj].F = self.meshes[ii][jj].F.to(device)
        for ii in range(self.nM):
            self.LCs[ii] = self.LCs[ii].to(device)
            for jj in range(self.nS - 1):
                self.hfList[ii][jj] = self.hfList[ii][jj].to(device)
                self.poolMats[ii][jj] = self.poolMats[ii][jj].to(device)
                self.dofs[ii][jj] = self.dofs[ii][jj].to(device)


class TestMeshes:
    def __init__(self, meshPathList, nSubd=2):
        """
        Inputs:
            meshPathList: list of paths to .obj files
        """
        self.meshes = preprocessTestShapes(meshPathList, nSubd, is_quad=True)  # Adjust preprocessing
        self.nM = len(self.meshes)  # number of meshes
        self.nS = len(self.meshes[0])  # number of subdivision levels

        # parameters
        self.hfList = None  # half flap index information
        self.poolMats = None  # vertex one-ring pooling matrices
        self.dofs = None  # vertex degrees of freedom
        self.LCs = None  # vector of differential coordinates

    def getInputData(self, mIdx):
        """
        get input data for the network
        Inputs: 
            mIdx: mesh index
        """
        input = torch.cat((
            self.meshes[mIdx][0].V,
            self.LCs[mIdx]),
            dim=1)
        return input  # (nV x Din)

    def computeLaplaceCoordinatesQuad(self,V, faces):
        """
        Compute Laplace coordinates for a quadmesh.
        
        Inputs:
            V: (N, 3) tensor of vertex positions
            faces: (F, 4) tensor of quad faces (vertex indices)
        
        Outputs:
            laplace_coords: (N, 3) tensor of Laplace coordinates
        """
        # Step 1: Generate directed edges
        edges = []
        for f in faces:
            edges.extend([
                [f[0], f[1]],  # Edge v0 -> v1
                [f[1], f[2]],  # Edge v1 -> v2
                [f[2], f[3]],  # Edge v2 -> v3
                [f[3], f[0]],  # Edge v3 -> v0
            ])
        edges = torch.tensor(edges, dtype=torch.long)
    
        # Step 2: Compute edge vectors
        edge_vectors = V[edges[:, 1]] - V[edges[:, 0]]  # Edge vectors v_j - v_i
    
        # Step 3: Accumulate edge vectors at each vertex
        # Create sparse pooling matrix to map edges to vertices
        num_vertices = V.shape[0]
        num_edges = edges.shape[0]
        row_indices = edges[:, 0]  # Start vertices (v_i)
        col_indices = torch.arange(num_edges)  # Edge indices
        values = torch.ones(num_edges, dtype=torch.float32)  # All edges contribute equally
        pooling_matrix = torch.sparse_coo_tensor(
            indices=torch.stack([row_indices, col_indices]),
            values=values,
            size=(num_vertices, num_edges),
        )
    
        # Pool edge vectors
        pooled_vectors = torch.sparse.mm(pooling_matrix, edge_vectors)
    
        # Step 4: Normalize by degree
        degree = pooling_matrix.sum(dim=1).to_dense()  # Degrees of each vertex
        laplace_coords = pooled_vectors / degree.unsqueeze(1)
    
        return laplace_coords

    def getHalfFlap(self):
        """
        Create a list of half flap information for quad-meshes.
        HF[meshIdx][subdIdx] = [v_blue, v_red, v_purple, v_yellow]
        """
        HF = [None] * self.nM
        for ii in range(self.nM):
            fifj = [None] * (self.nS - 1)
            for jj in range(self.nS - 1):
                idx = self.meshes[ii][jj].hfList[:, [0, 1, 2, 3]]
                fifj[jj] = idx.reshape(-1, 4)  # Adjust for quad-mesh indexing
                print(idx)
                print(" prinint idx ")
                input()
            HF[ii] = list(fifj)
        return HF

    # Other methods remain the same.
def preprocessTestShapes(meshPathList, nSubd, is_quad=False):
    meshes = []
    for path in meshPathList:
        V, F = load_mesh(path)  # Adjust to load quad-mesh
        FList, hfList = computeFlapList(V, F, numSubd=nSubd, is_quad=is_quad)
        meshes.append([
            {'V': V, 'F': F, 'hfList': hfList} for V, F, hfList in zip(FList, hfList)
        ])
    return meshes

def quadSubdivide(V, F):
    """
    Perform Catmull-Clark subdivision for quad-meshes.
    Inputs:
        V: (nV, 3) tensor of vertices
        F: (nF, 4) tensor of quad faces
    Outputs:
        VV: new vertex positions
        FF: new face indices
    """
    # Calculate face points
    face_points = V[F].mean(dim=1)

    # Calculate edge points
    edge_map = {}
    edge_points = []
    for face in F:
        edges = [(face[i].item(), face[(i + 1) % 4].item()) for i in range(4)]
        for e in edges:
            edge = tuple(sorted(e))
            if edge not in edge_map:
                edge_map[edge] = len(edge_points)
                edge_points.append(V[edge,:].mean(dim=0))
    edge_points = torch.stack(edge_points)

    # Calculate new vertices
    VV = torch.cat([V, face_points, edge_points])

    # Build new faces
    FF = []
    for f_idx, face in enumerate(F):
        face_center = len(V) + f_idx
        edges = [(face[i].item(), face[(i + 1) % 4].item()) for i in range(4)]
        k = tuple(sorted(edges[0]))
        edge_indices = [edge_map[tuple(sorted(e))] for e in edges]
        for i in range(4):
            FF.append([face[i], edge_indices[i], face_center, edge_indices[i - 1]])
    FF = torch.tensor(FF, dtype=torch.long)

    return VV, FF

def computeFlapList(V, F, numSubd=2, is_quad=False):
    """
    Compute half-flap lists for quad-meshes.
    """
    FList = []
    hfList = []

    for _ in range(numSubd):
        VV, FF = quadSubdivide(V, F)
        
        # Create half-flap data
        hf = []
        for f in F:
            for i in range(len(f)):
                v0 = f[i]
                v1 = f[(i + 1) % len(f)]
                v2 = f[(i + 2) % len(f)]
                v3 = f[(i + 3) % len(f)]
                hf.append([v0, v1, v2, v3])
        FList.append(FF)
        hfList.append(torch.tensor(hf, dtype=torch.long))

        V, F = VV, FF

    FList.append(F)
    hfList.append(None)
    return FList, hfList

