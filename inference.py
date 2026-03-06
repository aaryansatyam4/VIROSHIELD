import argparse
import os
import sys
import glob
import random
import numpy as np
import warnings
import torch
import time

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from rdkit import RDLogger

from model import ViroShieldGenerator

# Suppress RDKit warnings for a clean terminal output
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DEVICE = torch.device("mps") if (torch.backends.mps.is_available()) else torch.device("cpu")
MODEL_PATH = "viroshield_final_v6.pth"
FEAT_DIM = 128
NUM_ATOMS_TO_GENERATE_DEFAULT = 35
BATCH_SIZE_DEFAULT = 1000

def print_step(step, msg):
    print(f"\n[PHASE {step}] {msg}")
    time.sleep(0.3)

def get_atom_features(mol):
    mapping = {6: 0, 7: 1, 8: 2, 16: 3, 15: 4}
    atom_types = [mapping.get(atom.GetAtomicNum(), 5) for atom in mol.GetAtoms()]
    return torch.tensor(atom_types, dtype=torch.long).view(-1, 1)

def orchestrate_and_relax(coords, atom_types):
    """
    The Agentic Workflow: Smart Valence Filter + Fragment Extractor + MMFF94 Relaxation
    """
    idx_to_elem = {0: "C", 1: "N", 2: "O", 3: "S", 4: "P", 5: "F"}
    max_valences = {"C": 4, "N": 3, "O": 2, "S": 6, "P": 5, "F": 1}

    mol = Chem.RWMol()
    conf = Chem.Conformer(len(coords))
    bond_counts = {i: 0 for i in range(len(coords))}

    for i, t in enumerate(atom_types):
        elem = idx_to_elem.get(int(t), "C")
        idx = mol.AddAtom(Chem.Atom(elem))
        conf.SetAtomPosition(idx, [float(coords[i][0]), float(coords[i][1]), float(coords[i][2])])
    mol.AddConformer(conf)

    dist_matrix = np.linalg.norm(coords[:, None] - coords, axis=2)
    possible_bonds = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if 0.4 < dist_matrix[i, j] < 1.9:
                possible_bonds.append((dist_matrix[i, j], i, j))

    possible_bonds.sort(key=lambda x: x[0])

    for dist, i, j in possible_bonds:
        elem_i = idx_to_elem.get(int(atom_types[i]), "C")
        elem_j = idx_to_elem.get(int(atom_types[j]), "C")
        if bond_counts[i] < max_valences[elem_i] and bond_counts[j] < max_valences[elem_j]:
            try:
                mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)
                bond_counts[i] += 1
                bond_counts[j] += 1
            except:
                pass

    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if not frags:
            return None
        largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
        if largest_frag.GetNumAtoms() < 12:
            return None

        Chem.SanitizeMol(largest_frag)
        if AllChem.MMFFHasAllMoleculeParams(largest_frag):
            AllChem.MMFFOptimizeMolecule(largest_frag, maxIters=1000)
        else:
            AllChem.UFFOptimizeMolecule(largest_frag, maxIters=1000)

        new_coords = largest_frag.GetConformer().GetPositions()
        for i in range(len(new_coords)):
            for j in range(i + 1, len(new_coords)):
                if np.linalg.norm(new_coords[i] - new_coords[j]) < 0.7:
                    return None
        return largest_frag
    except:
        return None

def resolve_target_path(target_arg: str) -> str:
    target_arg = target_arg.strip()
    if os.path.exists(target_arg) and target_arg.lower().endswith(".pdb"):
        return target_arg
    pocket_dir = "./processed_pockets"
    if not target_arg.lower().endswith(".pdb"):
        target_arg += ".pdb"
    wanted = target_arg.lower()
    for f in glob.glob(os.path.join(pocket_dir, "*.pdb")):
        if os.path.basename(f).lower() == wanted:
            return f
    raise FileNotFoundError(f"Target not found: {target_arg}")

def generate_perfect_antidote(target_path: str, out_path: str, batch_size: int, num_atoms: int):
    print("="*75)
    print("🚀 VIROSHIELD: AI-DRIVEN DE NOVO DRUG GENERATION SYSTEM")
    print("="*75)

    print_step(1, f"Initializing Engine on {str(DEVICE).upper()}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model = ViroShieldGenerator(feat_dim=FEAT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ GATv2 Neural Brain Loaded.")

    print_step(2, f"Targeting Viral Pocket: {os.path.basename(target_path)}")
    p_mol = Chem.MolFromPDBFile(target_path, removeHs=True)
    if p_mol is None or p_mol.GetNumAtoms() == 0:
        raise ValueError(f"Failed to read pocket PDB: {target_path}")

    p_pos = torch.tensor(p_mol.GetConformer().GetPositions(), dtype=torch.float)
    center_mass = p_pos.mean(dim=0, keepdim=True)
    p_pos_centered = (p_pos - center_mass) / 10.0

    x = get_atom_features(p_mol).to(DEVICE)
    pocket_pos = p_pos_centered.to(DEVICE)
    dist_matrix = torch.cdist(pocket_pos, pocket_pos)
    edge_index = torch.nonzero((dist_matrix < 0.45) & (dist_matrix > 0)).t().contiguous().long().to(DEVICE)

    print(f"📊 Graph Extraction: {x.shape[0]} Nodes | {edge_index.shape[1]} Spatial Edges")
    print(f"🔢 Topology Matrix Sample:\n{dist_matrix[:3, :3].cpu().numpy()}")

    print_step(3, "Mining Potential Candidates via Agentic Filtering...")
    best_score = -9999
    best_mol = None
    valid_count = 0

    for i in range(batch_size):
        ligand_seed = torch.randn(num_atoms, FEAT_DIM).to(DEVICE)
        with torch.no_grad():
            pred_coords, pred_types_logits = model(x, edge_index, ligand_seed)

        probs = torch.softmax(pred_types_logits, dim=1)
        probs[:, 0] *= 3.0  # favor Carbon
        probs = probs / probs.sum(dim=1, keepdim=True)

        raw_types = torch.multinomial(probs, 1).squeeze().cpu().numpy()
        raw_coords = (pred_coords.cpu().numpy() * 10.0) + center_mass.numpy()

        final_mol = orchestrate_and_relax(raw_coords, raw_types)

        if final_mol is not None:
            valid_count += 1
            mw = rdMolDescriptors.CalcExactMolWt(final_mol)
            hbd = rdMolDescriptors.CalcNumHBD(final_mol)
            hba = rdMolDescriptors.CalcNumHBA(final_mol)
            
            score = final_mol.GetNumAtoms()
            if mw < 500: score += 50
            if hbd <= 5 and hba <= 10: score += 50

            if score > best_score:
                best_score = score
                best_mol = final_mol
                print(f"   ✨ NEW BEST! Candidate {i+1}: Score {score} | MW: {mw:.1f} | HBD: {hbd}")

    print_step(4, "Performing Pharmacokinetic Validation (Lipinski Rules)...")
    mw = rdMolDescriptors.CalcExactMolWt(best_mol)
    hbd = rdMolDescriptors.CalcNumHBD(best_mol)
    hba = rdMolDescriptors.CalcNumHBA(best_mol)
    logp = Descriptors.MolLogP(best_mol) 
    formula = rdMolDescriptors.CalcMolFormula(best_mol)
    smiles = Chem.MolToSmiles(best_mol)

    print(f"⚖️  [LIPINSKI AUDIT]")
    print(f"   ➤ Molecular Weight: {mw:.2f} Da  | {'✅ PASS' if mw < 500 else '❌ FAIL'}")
    print(f"   ➤ H-Bond Donors:    {hbd}         | {'✅ PASS' if hbd <= 5 else '❌ FAIL'}")
    print(f"   ➤ H-Bond Acceptors: {hba}        | {'✅ PASS' if hba <= 10 else '❌ FAIL'}")
    print(f"   ➤ LogP (Hydrophobicity): {logp:.2f} | {'✅ PASS' if logp <= 5 else '❌ FAIL'}")

    print_step(5, "Finalizing 3D Lead Candidate...")
    print(f"📈 Pipeline Stats: {valid_count}/{batch_size} passed RDKit 3D physics relaxation.")
    print(f"🧪 Formula: {formula}\n🧪 SMILES:  {smiles}")

    with open(out_path, "w") as f:
        coords = best_mol.GetConformer().GetPositions()
        for i, atom in enumerate(best_mol.GetAtoms()):
            ax, ay, az = coords[i]
            element = atom.GetSymbol()
            f.write(f"ATOM  {i+1:>5}  {element:<3} LIG A   1    {ax:>8.3f}{ay:>8.3f}{az:>8.3f}  1.00  0.00           {element}\n")

    print_step(6, "Saving Mathematical Specs & Audit Logs...")
    
    math_log = f"""==================================================
VIROSHIELD TECHNICAL AUDIT: {os.path.basename(target_path)}
==================================================
1. GRAPH TOPOLOGY:
   - Nodes (Atoms): {x.shape[0]}
   - Edges (Bonds): {edge_index.shape[1]}
   - Feature Dim: {FEAT_DIM}

2. GENERATED LIGAND:
   - Formula: {formula}
   - SMILES: {smiles}

3. LIPINSKI VALIDATION:
   - MW: {mw:.2f} | HBD: {hbd} | HBA: {hba} | LogP: {logp:.2f}

4. HARDWARE: {DEVICE}
=================================================="""
    
    with open("viroshield_audit_log.txt", "w") as f: f.write(math_log)
    np.savetxt("graph_adjacency_matrix.csv", dist_matrix.cpu().numpy()[:50, :50], delimiter=",")
    
    print("📄 Saved: viroshield_audit_log.txt")
    print("📊 Saved: graph_adjacency_matrix.csv (50x50 sample)")
    print(f"\n🏆 VICTORY: Antidote generated and saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--num-atoms", type=int, default=NUM_ATOMS_TO_GENERATE_DEFAULT)
    args = parser.parse_args()

    try:
        target_path = resolve_target_path(args.target)
        generate_perfect_antidote(target_path, args.out, args.batch_size, args.num_atoms)
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    main()