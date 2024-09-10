from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np

# 使用in要比not in快，因为not in在in之后还要取反 
element_set = {'Br','I','Cl','S','O','P','N','H','F','C','Si'}


def filter_yq(smi_list:list)->tuple:
    None_mol, repeat, element, logp_idx, big_mass, ionic = [], [], [], [], [], []
    smi_set = {} # 用来记录各个SMILES的数量 
    qualified_mols = [] # 返回的符合条件的分子
    qualified_smi = [] # 返回的符合条件的SMILES

    for i in range(len(smi_list)):
        quali = True
        mol = Chem.MolFromSmiles(smi_list[i])
        if mol is None:
            None_mol.append(i)
            continue
        s = smi_list[i]
        # repeat
        if s in smi_set:
            smi_set[s] += 1
            repeat.append(i)
            continue
        else:
            smi_set[s] = 1
        # ionic
        if '.' in s:
            ionic.append(i)
            continue
        # logp
        logp = Descriptors.MolLogP(mol) 
        if logp < -12.0 or logp > 24.0:
            logp_idx.append(i)
            continue
        # mass
        mw = Descriptors.ExactMolWt(mol)
        if  mw > 1000.0:
            big_mass.append(i)
            continue
        # element 放在最后，防止重复计算
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        for a in atoms:
            if a in element_set:
                continue
            else:
                element.append(i)
                quali = False
                break 
        if quali:
            mol.SetProp('SMILES', s)
            qualified_mols.append(mol)
            qualified_smi.append(s)
        # continue保证了各个列表没有交集
    unqualified = None_mol + repeat + element + logp_idx + big_mass + ionic
    print('unqualified:%d' % len(unqualified)) 
    return qualified_mols, unqualified, set(qualified_smi)    


# NIST
def filter_nist(smi_list:list)->tuple:
    repeat, element, logp_idx, big_mass, ionic = [], [], [], [], []
    nist_smi = {} # 用来记录各个SMILES的数量 
    qualified_mols = [] # 返回的符合条件的分子
    qualified_smi = [] # 返回的符合条件的SMILES

    for i in range(len(smi_list)):
        quali = True
        mol = Chem.MolFromSmiles(smi_list[i])
        # s = Chem.MolToSmiles(mol)
        s = smi_list[i]
        # repeat
        if s in nist_smi:
            nist_smi[s] += 1
            repeat.append(i)
            continue
        else:
            nist_smi[s] = 1
        # ionic
        if '.' in s:
            ionic.append(i)
            continue
        # logp
        logp = Descriptors.MolLogP(mol) 
        if logp < -12.0 or logp > 24.0:
            logp_idx.append(i)
            continue
        # mass
        mw = Descriptors.ExactMolWt(mol)
        if  mw > 1000.0:
            big_mass.append(i)
            continue
        # element 放在最后，防止重复计算
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        for a in atoms:
            if a in element_set:
                continue
            else:
                element.append(i)
                quali = False
                break 
        if quali:
            mol.SetProp('SMILES', s)
            qualified_mols.append(mol)
            qualified_smi.append(s)
        # continue保证了各个列表没有交集
    unqualified = repeat + element + logp_idx + big_mass + ionic
    # val/test/train : 102/99/4286
    print('unqualified:%d' % len(unqualified)) 
    # val/test/train : 11497/11501/232822
    return qualified_mols, unqualified, set(qualified_smi)    
    # 总255820


# Human Metabolome Database
def filter_dedup_save_hmdb(mols:list, smi_sets_list:list, save_path:str)->tuple:
    # [H+]，id为HMDB0059597
    # id为HMDB0258804的分子的SMILES错误，但Mol正确，可以MolToSmiles
    None_mol, repeat, element ,logp_idx, big_mass, ionic = [], [], [], [], [], []
    hmdb_smi = {}
    qualified_mols = []
    qualified_smi = []
    writer = AllChem.SDWriter(save_path)

    for i in range(len(mols)):    
        if mols[i] is None:
            None_mol.append(i)
            continue
        quali = True
        smi = Chem.MolToSmiles(mols[i])
        # repeat
        for smi_set in smi_sets_list:
            if smi in smi_set: # 数据集之间去重 
                repeat.append(i)
                quali = False
                break
        if not quali:
            continue
        if smi in hmdb_smi: # 数据集内部去重
            hmdb_smi[smi] += 1
            repeat.append(i)
            continue
        else:
            hmdb_smi[smi] = 1
        # ionic
        if '.' in smi:
            ionic.append(i)
            continue
        # logp 优先使用提供的logp，因为rdkit误差大
        logps = []
        try:
            logps.append(float(mols[i].GetProp('ALOGPS_LOGP')))
        except:
            pass
        try:
            logps.append(float(mols[i].GetProp('JCHEM_LOGP'))) 
        except:
            pass
        logps.append(Descriptors.MolLogP(mols[i]))
        logp = np.mean(logps)
        if logp < -12.0 or logp > 24.0:
            logp_idx.append(i)
            continue
        # mass
        try:
            mw = float(mols[i].GetProp('MOLECULAR_WEIGHT'))
        except:
            mw = Descriptors.ExactMolWt(mols[i]) # YQ不采用M_W或E_M
        if  mw > 1000.0:
            big_mass.append(i)
            continue
        # element 放在最后，防止重复计算
        atoms = [atom.GetSymbol() for atom in mols[i].GetAtoms()]
        for a in atoms:
            if a in element_set:
                continue
            else:
                element.append(i)
                quali = False
                break
        if quali:
            mols[i].SetProp('SMILES', smi)
            qualified_mols.append(mols[i])
            qualified_smi.append(smi)
            writer.write(mols[i])
    writer.close()
    unqualified = None_mol + repeat + element + logp_idx + big_mass + ionic
    print('unqualified:%d' % len(unqualified)) # 6794
    # nist->hmdb:118769
    # nist->chembel->hmdb: 118470
    return qualified_mols, unqualified, set(qualified_smi)


# Chembl
def filter_dedup_save_chembl(mols:list, smi_sets_list:list, save_path:str)->tuple:
    # CHEMBL4080644 MolToSmiles可以，MolFromSmiles不行
    None_mol, repeat, element, logp_idx, big_mass, ionic = [], [], [], [], [], []
    chembl_smi = {}
    qualified_mols = []
    qualified_smi = []
    writer = AllChem.SDWriter(save_path)

    for i in range(len(mols)):    
        if mols[i] is None:
            None_mol.append(i)
            continue
        smi = Chem.MolToSmiles(mols[i])
        quali = True
        # repeat
        for smi_set in smi_sets_list:
            if smi in smi_set: # 数据集之间去重 
                repeat.append(i)
                quali = False
                break
        if not quali:
            continue
        if smi in chembl_smi:
            chembl_smi[smi] += 1
            repeat.append(i)
            continue
        else:
            chembl_smi[smi] = 1
        # logp
        logp = Descriptors.MolLogP(mols[i])
        if logp < -12.0 or logp > 24.0:
            logp_idx.append(i)
            continue
        # ionic
        if '.' in smi:
            ionic.append(i)
            continue
        # mass
        if Descriptors.ExactMolWt(mols[i]) > 1000.0:
            big_mass.append(i)
            continue
        # element 放在最后，防止重复计算
        atoms = [atom.GetSymbol() for atom in mols[i].GetAtoms()]
        for a in atoms:
            if a in element_set:
                continue
            else:
                element.append(i)
                quali = False
                break
        if quali:
            mols[i].SetProp('SMILES', smi)
            qualified_mols.append(mols[i])
            qualified_smi.append(smi)
            writer.write(mols[i])
          
    writer.close()
    unqualified = None_mol + repeat + element + logp_idx + big_mass + ionic
    print('unqualified:%d' % len(unqualified)) # 
    return qualified_mols, unqualified, set(qualified_smi) # 


# 对mgf格式的质谱进行内部去重  
def filter_mgf(mol_list:list)->tuple:
    repeat, element, logp_idx, big_mass, ionic = [], [], [], [], []
    nist_smi = {} # 用来记录各个SMILES的数量 
    qualified_mols = [] # 返回的符合条件的分子
    qualified_smi = [] # 返回的符合条件的SMILES

    for i in range(len(mol_list)):
        quali = True
        s = mol_list[i].metadata['smiles']
        mol = Chem.MolFromSmiles(s)
        # repeat
        if s in nist_smi:
            nist_smi[s] += 1
            repeat.append(i)
            continue
        else:
            nist_smi[s] = 1
        # ionic
        if '.' in s:
            ionic.append(i)
            continue
        # logp
        try:
            logp = Descriptors.MolLogP(mol) 
        except:
            logp_idx.append(i)
            continue
        if logp < -12.0 or logp > 24.0:
            logp_idx.append(i)
            continue
        # mass
        mw = float(mol_list[i].metadata['exactmass'])
        if  mw > 1000.0:
            big_mass.append(i)
            continue
        # element 放在最后，防止重复计算
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        for a in atoms:
            if a in element_set:
                continue
            else:
                element.append(i)
                quali = False
                break 
        if quali:
            qualified_mols.append(mol_list[i])
            qualified_smi.append(s)
        # continue保证了各个列表没有交集
    unqualified = repeat + element + logp_idx + big_mass + ionic
    print('unqualified:%d' % len(unqualified)) 
    return qualified_mols, unqualified, set(qualified_smi)    