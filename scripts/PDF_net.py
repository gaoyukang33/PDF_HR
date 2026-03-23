import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def get_g1_parent_mapping():
    """Defines and returns the G1 parent mapping as a comma-separated string."""
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

    mapping[15] = 14 
    mapping[16] = 15 
    mapping[17] = 16 
    mapping[18] = 17 
    mapping[19] = 18
    mapping[20] = 19 
    mapping[21] = 20 

    mapping[22] = 14 
    mapping[23] = 22
    mapping[24] = 23
    mapping[25] = 24 
    mapping[26] = 25 
    mapping[27] = 26
    mapping[28] = 27

    return ",".join([str(x) for x in mapping])
class DFNet(nn.Module):
    def __init__(self, opt, batch_size=4, use_gpu=0, layer='UpperClothes', weight_norm=True, activation='relu', dropout=0.3, output_layer=None):
        super().__init__()
        input_size = opt['in_dim']
        hid_layer = [int(val) for val in opt['dims'].split(',')]
        output_size = opt['output_size']
        dims = [input_size] + hid_layer + [output_size]

        self.num_layers = len(dims) - 1
        
        for l in range(self.num_layers):
            setattr(self, f"lin{l}", nn.Linear(dims[l], dims[l + 1]))

        if opt['act'] == 'lrelu':
            self.actv = nn.LeakyReLU()
            self.out_actv = nn.ReLU()
        elif opt['act'] == 'relu':
            self.actv = nn.ReLU()
            self.out_actv = nn.ReLU()
        elif opt['act'] == 'softplus':
            self.actv = nn.Softplus(beta=opt['beta'])
            self.out_actv = nn.Softplus(beta=opt['beta'])

    def forward(self, p):
        x = p.reshape(p.size(0), -1)
        
        for l in range(self.num_layers):
            layer = getattr(self, f"lin{l}")
            x = layer(x)
            if l < self.num_layers - 1:
                x = self.actv(x)

        return self.out_actv(x)

class BoneMLP(nn.Module):
    def __init__(self, bone_dim, bone_feature_dim, parent=-1, act='relu', beta=100.):
        super().__init__()
        in_features = bone_dim if parent == -1 else bone_dim + bone_feature_dim
        n_features = bone_dim + bone_feature_dim

        if act == 'relu':
            act_layer = nn.ReLU()
        elif act == 'lrelu':
            act_layer = nn.LeakyReLU()
        elif act == 'softplus':
            act_layer = nn.Softplus(beta=beta)
        else:
            raise ValueError(f"Unsupported activation: {act}")

        self.net = nn.Sequential(
            nn.Linear(in_features, n_features),
            act_layer,
            nn.Linear(n_features, bone_feature_dim),
            act_layer
        )

    def forward(self, bone_feat):
        return self.net(bone_feat)

class StructureEncoder(nn.Module):
    def __init__(self, opt, local_feature_size=6):
        super().__init__()
        self.bone_dim = 4  
        self.input_dim = self.bone_dim 
        self.parent_mapping = [int(val) for val in opt['smpl_mapping'].split(',')]
        self.num_joints = len(self.parent_mapping)
        self.out_dim = self.num_joints * local_feature_size

        self.net = nn.ModuleList([
            BoneMLP(self.input_dim, local_feature_size, self.parent_mapping[i], opt['act'], opt['beta']) 
            for i in range(self.num_joints)
        ])

    def get_out_dim(self):
        return self.out_dim

    def forward(self, quat):
        features = [None] * self.num_joints
        for i, mlp in enumerate(self.net):
            parent = self.parent_mapping[i]
            if parent == -1:
                features[i] = mlp(quat[:, i, :])
            else:
                inp = torch.cat((quat[:, i, :], features[parent]), dim=-1)
                features[i] = mlp(inp)
                
        return torch.cat(features, dim=-1) 

class StructureEncoder1D(StructureEncoder):
    def __init__(self, opt, local_feature_size=6):
        nn.Module.__init__(self) 
        self.bone_dim = 1  
        self.input_dim = self.bone_dim 
        
        self.parent_mapping = [int(val) for val in opt['smpl_mapping'].split(',')]
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

        self.net = nn.ModuleList([
            BoneMLP(self.input_dim, local_feature_size, self.parent_mapping[i], opt['act'], opt['beta']) 
            for i in range(self.num_joints)
        ])

    def forward(self, x):
        features = [None] * self.num_joints
        
        for i in self.topo_order:
            mlp = self.net[i]           
            parent = self.parent_mapping[i]
            
            if parent == -1:
                features[i] = mlp(x[:, i, :])
            else:
                inp = torch.cat((x[:, i, :], features[parent]), dim=-1)
                features[i] = mlp(inp)
                
        return torch.cat(features, dim=-1)

class PDFHR_Adapter(nn.Module):
    def __init__(self, device):
        super().__init__()
        
        local_feature_size = 16  
        self.opt = {
            'smpl_mapping': get_g1_parent_mapping(),
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
        x_reshaped = x.unsqueeze(-1)
        latent = self.enc(x_reshaped) 
        dist = self.dfnet(latent)     
        return dist.squeeze(-1) 

def train_pdfhr_on_g1(
    dataset,
    val_dataset=None,
    batch_size=1024 * 1024,  
    epochs=20,               
    lr=1e-3,                 
    num_workers=0,
    device=None,
    save_dir="./prior_ckpts"
):
    print("Initializing PDF-HR Adapter for G1...")
    model = PDFHR_Adapter(device=device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() 

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False) if val_dataset else None

    def run_epoch(loader, is_train: bool):
        model.train(is_train)
        total_loss, total_n = 0.0, 0
        pbar = tqdm(loader, desc="Train" if is_train else "Val")
        
        for x, y in pbar:
            if is_train:
                opt.zero_grad(set_to_none=True) 
                pred = model(x) 
                loss = loss_fn(pred, y)
                loss.backward()
                opt.step()
            else:
                with torch.no_grad():
                    pred = model(x)
                    loss = loss_fn(pred, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
        return total_loss / max(1, total_n)

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, is_train=True)
        val_info = f"| val {run_epoch(val_loader, is_train=False):.6f}" if val_loader else ""
            
        print(f"[{epoch:03d}/{epochs}] train {train_loss:.6f} {val_info}")

        os.makedirs(save_dir, exist_ok=True) 
        ckpt_path = os.path.join(save_dir, f"pdfhr_model_epoch_{epoch}.pt")
        torch.save({"model": model.state_dict()}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    return model

if __name__ == "__main__":
    pre_compute_pt = './precomputed_data/sampling_pose_L1.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset")
    data = torch.load(pre_compute_pt, map_location=device) 
    inputs = data['db'].float()
    targets = data['dis'].float().squeeze()

    # Shuffle dataset
    perm = torch.randperm(inputs.shape[0], device=device)
    inputs, targets = inputs[perm], targets[perm]

    train_ds = TensorDataset(inputs.contiguous(), targets.contiguous())
    print(f"Train set: {len(train_ds)}")

    model = train_pdfhr_on_g1(
        train_ds,
        val_dataset=None,
        batch_size=1024 * 64, 
        epochs=5,
        lr=1e-3, 
        save_dir="./prior_ckpts",
        device=device
    )