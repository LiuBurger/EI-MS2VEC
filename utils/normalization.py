import numpy as np
from matchms import Spectrum
from matchms.exporting import save_as_mgf
from rdkit.Chem import Descriptors


def norm_pre_sdf2mgf(mols:list, save_path:str)->list: # 预测谱的正则化
    spectra = []
    for mol in mols:
        mzs_intens = mol.GetProp('PREDICTED SPECTRUM').split('\n')
        mzs = [float(mz_inten.split(' ')[0]) for mz_inten in mzs_intens]
        intens = [float(mz_inten.split(' ')[1]) for mz_inten in mzs_intens]
        mzs = np.array(mzs)
        intens = np.array(intens)/max(intens)
        mw = Descriptors.ExactMolWt(mol)
        spectrum = Spectrum(mzs, intens, metadata={'SMILES': mol.GetProp('SMILES'),
                                                         'MW': mw})
        spectra.append(spectrum)
    if save_path is not None:
        save_as_mgf(spectra, save_path)
    
    return spectra


def normalization_mgf2mgf(mols:list, save_path:str=None)->list:
    spectra = []
    for mol in mols:
        mzs = mol.mz
        intens = mol.intensities
        intens = intens/max(intens)
        spectrum = Spectrum(mzs, intens, metadata={'smiles': mol.metadata['smiles']})
        spectra.append(spectrum)
    if save_path is not None:
        save_as_mgf(spectra, save_path)
    return spectra