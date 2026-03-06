import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ViroShieldGenerator(nn.Module):
    def __init__(self, feat_dim=128): 
        super().__init__()
        
        
        self.embedding = nn.Embedding(6, feat_dim)
        
        
        self.gat1 = GATv2Conv(feat_dim, feat_dim, heads=4, concat=False)
        self.bn1 = nn.BatchNorm1d(feat_dim)
        self.gat2 = GATv2Conv(feat_dim, feat_dim, heads=4, concat=False)
        self.bn2 = nn.BatchNorm1d(feat_dim)
        
        
        self.ligand_self_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=4, batch_first=True)
        self.norm_ligand = nn.LayerNorm(feat_dim)

        
        self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(feat_dim)
        
        
        self.coord_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.SiLU(),  
            nn.Linear(feat_dim * 2, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, 3) 
        )
        
        
        self.type_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.SiLU(),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, 6) 
        )

    def forward(self, pocket_x, pocket_edge_index, ligand_seed):
        
        x = self.embedding(pocket_x.long()).squeeze(1)
        x_in = x
        x = self.gat1(x, pocket_edge_index)
        x = self.bn1(x)
        x = F.elu(x) + x_in 
        
        x_in = x
        x = self.gat2(x, pocket_edge_index)
        x = self.bn2(x)
        x = F.elu(x) + x_in
        
        seed_seq = ligand_seed.unsqueeze(0)
        self_out, _ = self.ligand_self_attn(seed_seq, seed_seq, seed_seq)
        ligand_feat = self.norm_ligand(seed_seq + self_out)
        
        
        pocket_seq = x.unsqueeze(0) 
        attn_output, _ = self.cross_attn(ligand_feat, pocket_seq, pocket_seq)
        final_feat = self.attn_norm(attn_output + ligand_feat).squeeze(0)
        
        
        predicted_coords = self.coord_mlp(final_feat)
        predicted_types = self.type_mlp(final_feat)
        
        return predicted_coords, predicted_types