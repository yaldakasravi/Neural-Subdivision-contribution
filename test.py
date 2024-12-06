from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from include import *
from models import *
NETPARAMS = 'netparams.dat'

# run the file as "python test.py ./path/to/folder/ ./path/to/testMesh.obj"
# WARNING
# - the current code only works for .obj file without boundaries :P 
# - there are two failure cases mentioned in the paper:
#   1. testing triangles has aspect ratios that are not in the training data
#   2. testing triangles has triangle size that are not in the training data
#   (we include a failure case in the data: the ear of the horse.obj)

def main():

    # load hyper parameters
    folder = sys.argv[1]
    with open(folder + 'hyperparameters.json', 'r') as f:
        params = json.load(f)
    params['numSubd'] = 2 # number of subdivision levels at test time

    print(os.path.basename(sys.argv[2]))

    # load validation set
    meshPath = [sys.argv[2]]
    T = TestMeshes(meshPath, params['numSubd'])
    T.computeParameters()
    if not torch.cuda.is_available():
        params['device'] = 'cpu'
    T.toDevice(params["device"])

    # initialize network 
    net = SubdNet(params)
    net = net.to(params['device'])
    net.load_state_dict(torch.load(params['output_path'] + NETPARAMS, map_location=torch.device(params["device"])))
    net.eval()

    # write output shapes (test set)
    mIdx = 0
    scale = 1.0 # may need to adjust the scale of the mesh since the network is not scale invariant
    meshName = os.path.basename(sys.argv[2])[:-4] # meshName: "bunny"

    x = T.getInputData(mIdx)
    curves = T.compute_curve()
    outputs = net(x, mIdx,T.hfList,T.poolMats,T.dofs,curves) 

    for ii in range(len(outputs)):
        x = outputs[ii].cpu() * scale
        tgp.writeOBJ(params['output_path'] + meshName + '_subd' + str(ii) + '.obj',x, T.meshes[mIdx][ii].F.to('cpu'))

    # write rotated output shapes
    x = T.getInputData(mIdx)
    curves = T.compute_curve()

    dV = torch.rand(1,3).to(params['device'])
    R = random3DRotation().to(params['device'])
    x[:,:3] = x[:,:3].mm(R.t())
    x[:,3:] = x[:,3:].mm(R.t())
    x[:,:3] += dV
    outputs = net(x, mIdx,T.hfList,T.poolMats,T.dofs,curves)

    for ii in range(len(outputs)):
        x = outputs[ii].cpu() * scale
        tgp.writeOBJ(params['output_path'] + meshName + '_rot_subd' + str(ii) + '.obj',x, T.meshes[mIdx][ii].F.to('cpu'))


if __name__ == '__main__':
    main()
