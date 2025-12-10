import torch
import torch.nn as nn


# define the mapping tree
def get_g1_parent_mapping():
    mapping = [-1] * 29

    mapping[12] = -1 
    mapping[13] = 12 
    mapping[14] = 13 

    mapping[0] = 12 
    mapping[1] = 0  
    mapping[2] = 1  
    mapping[3] = 2  
    mapping[4] = 3  
    mapping[5] = 4  

    mapping[6] = 12
    mapping[7] = 6  
    mapping[8] = 7  
    mapping[9] = 8  
    mapping[10] = 9 
    mapping[11] = 10

    mapping[15] = 14 # L_Shoulder_Pitch -> Waist_Pitch
    mapping[16] = 15 # Roll -> Pitch
    mapping[17] = 16 # Yaw -> Roll
    mapping[18] = 17 # Elbow -> Shoulder_Yaw
    mapping[19] = 18 # Wrist_Roll -> Elbow
    mapping[20] = 19 # Wrist_Pitch -> Roll
    mapping[21] = 20 # Wrist_Yaw -> Pitch

    mapping[22] = 14 # R_Shoulder_Pitch -> Waist_Pitch
    mapping[23] = 22
    mapping[24] = 23
    mapping[25] = 24 # Elbow
    mapping[26] = 25 # Wrist
    mapping[27] = 26
    mapping[28] = 27

    return ",".join([str(x) for x in mapping])



class DFNet(nn.Module):

    def __init__(self, opt, batch_size=4, use_gpu=0, layer='UpperClothes', weight_norm=True, activation='relu',
                 dropout=0.3, output_layer=None):
        super().__init__()
        input_size = opt['in_dim']
        hid_layer = opt['dims'].split(',')
        hid_layer = [int(val) for val in hid_layer]
        output_size = opt['output_size']
        dims = [input_size] + [d_hidden for d_hidden in hid_layer] + [output_size]

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            # if weight_norm:
            #     lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        if opt['act'] == 'lrelu':
            self.actv = nn.LeakyReLU()
            self.out_actv = nn.ReLU()

        if opt['act'] == 'relu':
            self.actv = nn.ReLU()
            self.out_actv = nn.ReLU()

        if opt['act'] == 'softplus':
            self.actv = nn.Softplus(beta=opt['beta'])
            self.out_actv = nn.Softplus(beta=opt['beta'])

    def forward(self, p):

        x = p.reshape(len(p), -1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.actv(x)

        x = self.out_actv(x)

        return x


class BoneMLP(nn.Module):
    """from LEAP code(CVPR21, Marko et al)"""

    def __init__(self, bone_dim, bone_feature_dim, parent=-1, act='relu', beta=100.):
        super(BoneMLP, self).__init__()
        if parent == -1:
            in_features = bone_dim
        else:
            in_features = bone_dim + bone_feature_dim
        n_features = bone_dim + bone_feature_dim

        if act == 'relu':
            self.net = nn.Sequential(
                nn.Linear(in_features, n_features),
                nn.ReLU(),
                nn.Linear(n_features, bone_feature_dim),
                nn.ReLU()
            )

        if act == 'lrelu':
            self.net = nn.Sequential(
                nn.Linear(in_features, n_features),
                nn.LeakyReLU(),
                nn.Linear(n_features, bone_feature_dim),
                nn.LeakyReLU()
            )
        if act == 'softplus':
            self.net = nn.Sequential(
                nn.Linear(in_features, n_features),
                nn.Softplus(beta=beta),
                nn.Linear(n_features, bone_feature_dim),
                nn.Softplus(beta=beta)
            )

    def forward(self, bone_feat):

        return self.net(bone_feat)


class StructureEncoder(nn.Module):
    """from LEAP code(CVPR21, Marko et al)"""

    def __init__(self, opt, local_feature_size=6):
        super().__init__()

        self.bone_dim = 4  # 3x3 for pose and 1x3 for joint loc  #todo: change this encodibg for quaternion
        self.input_dim = self.bone_dim  # +1 for bone length
        # self.parent_mapping = get_parent_mapping('smpl')
        smpl_mapping = opt['smpl_mapping'].split(',')
        self.parent_mapping = [int(val) for val in smpl_mapping]

        self.num_joints = len(self.parent_mapping)
        self.out_dim = self.num_joints * local_feature_size

        self.net = nn.ModuleList(
            [BoneMLP(self.input_dim, local_feature_size, self.parent_mapping[i], opt['act'], opt['beta']) for i in
             range(self.num_joints)])

    def get_out_dim(self):
        return self.out_dim

    @classmethod
    def from_cfg(cls, config):
        return cls(
            local_feature_size=config['local_feature_size'],
            parent_mapping=config['parent_mapping']
        )

    def forward(self, quat):
        """
        Args:
            pose: B x num_joints x 4
            rel_joints: B x num_joints x 3
        """

        # fwd pass through the bone encoder
        features = [None] * self.num_joints
        for i, mlp in enumerate(self.net):
            parent = self.parent_mapping[i]
            if parent == -1:
                features[i] = mlp(quat[:, i, :])
            else:
                inp = torch.cat((quat[:, i, :], features[parent]), dim=-1)
                features[i] = mlp(inp)
        features = torch.cat(features, dim=-1)  # B x f_len
        return features



class StructureEncoder1D(StructureEncoder):

    def __init__(self, opt, local_feature_size=6):
        
        nn.Module.__init__(self) 
        self.bone_dim = 1  
        self.input_dim = self.bone_dim 
        
        smpl_mapping = opt['smpl_mapping'].split(',')
        self.parent_mapping = [int(val) for val in smpl_mapping]
        self.num_joints = len(self.parent_mapping)
        self.out_dim = self.num_joints * local_feature_size

        children_map = {i: [] for i in range(self.num_joints)}
        roots = []
        for i, parent in enumerate(self.parent_mapping):
            if parent == -1:
                roots.append(i)
            else:
                children_map[parent].append(i)
        
        self.topo_order = []
        queue = roots[:]
        while queue:
            curr = queue.pop(0)
            self.topo_order.append(curr)
            children = sorted(children_map[curr])
            queue.extend(children)
            
        if len(self.topo_order) != self.num_joints:
            print(f"Warning: Topological sort missed some joints. Check mapping! Found {len(self.topo_order)}/{self.num_joints}")



        self.net = nn.ModuleList(
            [BoneMLP(self.input_dim, local_feature_size, self.parent_mapping[i], opt['act'], opt['beta']) for i in
             range(self.num_joints)])

    def forward(self, x):
        features = [None] * self.num_joints
        
        for i in self.topo_order:
            mlp = self.net[i]           
            parent = self.parent_mapping[i]
            
            if parent == -1:
                features[i] = mlp(x[:, i, None])
            else:
                inp = torch.cat((x[:, i, None], features[parent]), dim=-1)
                
                features[i] = mlp(inp)
                
        features = torch.cat(features, dim=-1)  # B x f_len
        return features

class NRDF_Adapter(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        
        parent_map_str = get_g1_parent_mapping()
        local_feature_size = 16  
        
        self.opt = {
            'smpl_mapping': parent_map_str,
            'local_feature_size': local_feature_size,
            'in_dim': 29 * local_feature_size, 
            'dims': '512,256,128',
            'output_size': 1,
            'act': 'softplus',               
            'beta': 100.0,
            'weight_norm': True
        }
        
        self.enc = StructureEncoder1D(self.opt, local_feature_size=local_feature_size)
        
        self.dfnet = DFNet(self.opt)
        
        self.to(device)

    def forward(self, x):

        latent = self.enc(x) # (B, 29*16)
        
        dist = self.dfnet(latent)     # (B, 1)

        return dist.squeeze(-1) # (B, )