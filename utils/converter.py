import re
import pandas as pd
import numpy as np
from matchms import Spectrum, set_matchms_logger_level
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
from rdkit import Chem
set_matchms_logger_level("ERROR")


# 用于让NIST17可以被rdkit读取
def correct_format(file_path:str, save_path:str):
    fr = open(file_path, 'r')
    lines = fr.readlines()
    fr.close()
    print(f'Num of lines before modification: {len(lines)}')
    new_lines = []
    new_lines.append(lines[0])
    for i in range(1, len(lines)):
        if lines[i].startswith('>  <NAME>') and not lines[i-1].startswith('M  END'):
            new_lines.append('M  END\n')
        new_lines.append(lines[i])
    print(f'Num of lines after modification: {len(new_lines)}')
    fw = open(save_path, 'w')
    fw.writelines(new_lines)
    fw.close()


def NIST17sdf2mgf(file_path:str, save_path:str):
    suppl = Chem.SDMolSupplier(file_path)
    mols = [mol for mol in suppl if mol]
    spectra = []
    for mol in mols:
        nistno = mol.GetProp('NISTNO')
        smi = mol.GetProp('SMILES')
        inchikey = mol.GetProp('INCHIKEY')
        formula = mol.GetProp('FORMULA')
        mw = int(mol.GetProp('MW'))
        ex_mass = float(mol.GetProp('EXACT MASS'))
        num_peaks = int(mol.GetProp('NUM PEAKS'))
        mzs_intens = mol.GetProp('MASS SPECTRAL PEAKS').split('\n')
        mzs_intens = [(float(mz_inten.split(' ')[0]), float(mz_inten.split(' ')[1])) 
                    for mz_inten in mzs_intens]
        mzs_intens = sorted(mzs_intens, key=lambda x: x[0])
        mzs, intens = zip(*mzs_intens)
        mzs = np.array(mzs)
        intens = np.array(intens) / max(intens)
        spectrum = Spectrum(mzs, intens, metadata=
                            {'nistno': nistno, 'smiles': smi, 'inchikey': inchikey,'formula': formula, 
                             'mw': mw, 'EXACT MASS': ex_mass, 'NUM PEAKS': num_peaks})
        # spectrum.set('mw', mw)
        # spectrum.set('EXACT MASS', ex_mass)
        # spectrum.set('NUM PEAKS', num_peaks)
        spectra.append(spectrum)
    save_as_mgf(spectra, save_path)


def extract_number(filename:str):
    match = re.match(r'(\d+)', filename)
    return int(match.group()) if match else filename


def gen_spectrums_from_csv(file_list:list, save:bool=False, save_path:str='data/meassured_spectra.mgf'):
    spectra = []
    for i in range(len(file_list)):
        f = 'data/extra_test_set/' + file_list[i]
        data = pd.read_csv(f, header=None)
        mz = np.array(round(data[0].astype(float)))
        inten = np.array(data[1].astype(float))
        inten /= max(inten) # 归一化
        #delete noise
        keep = np.where(inten > 0.001)[0] # 过滤掉强度小于0.001的峰
        mz = mz[keep]
        inten = inten[keep]
        if max(mz)>1000:
            continue
        else:                
            spectrum = Spectrum(mz=mz, intensities=inten,
                                metadata={'compound_name': 'substance_measured_'+str(file_list[i])})
            spectra.append(spectrum)
    if save:
        save_as_mgf(spectra, save_path) 
    return spectra


# import sqlite3
# gradedb1 = sqlite3.connect("data/author/The_expanded_in-silico_library.db")
# cursor1 = gradedb1.cursor()
# content1 = cursor1.execute("SELECT COMPID, SMILES, MZS, INTENSITYS, EXACTMOLWT from IN_SILICO_LIBRARY").fetchall()
# print(len(content1))
# import tqdm

# spectrums1 = []
# for c1 in tqdm.tqdm(content1):
#     compid = c1[0]
#     smiles = c1[1]
#     mzs_str = c1[2]
#     mzs = re.findall(r'\d+', mzs_str)
#     mzs = np.array([float(mz) for mz in mzs])
#     intensitys_str = c1[3]
#     intensitys = re.findall(r'\d+\.\d+', intensitys_str)
#     intensitys = np.array([float(intensity) for intensity in intensitys])
#     mw = c1[4]
#     s = Spectrum(mzs, intensitys, metadata={
#         'COMPID': compid, 'SMILES': smiles, 'MW':mw
#     })
#     spectrums1.append(s)
# print(len(spectrums1))
# save_as_mgf(spectrums1, './data/author/The_expanded_in-silico_library.mgf')


def read_msp2mgf(file_path:str, save_path:str=None):
    f = open(file_path, 'r')
    mols = f.read().split('\n\n')[:-1]
    spectra = []
    for mol in mols:
        lines = mol.split('\n')
        Name, DB, InChIKey, SMILES, Precursor_type, Spectrum_type, PrecursorMZ, ExactMass, Num_Peaks =\
            None, None, None, None, None, None, None, None, None
        correct = True # 格式正确 
        mz, inten = [], []
        for l in lines:
            if l.startswith('Name:'):
                Name = l.split(': ')[-1]
            elif l.startswith('DB:'):
                DB = l.split(': ')[-1]
            elif l.startswith('InChIKey:'):
                InChIKey = l.split(': ')[-1]
            elif l.startswith('SMILES:'):
                SMILES = l.split(': ')[-1]
                if SMILES == 'NA':
                    correct = False
                    break # 格式不对直接跳出，没必要继续做 
            elif l.startswith('Precursor_type:'):
                Precursor_type = l.split(': ')[-1]
            elif l.startswith('Spectrum_type:'):
                Spectrum_type = l.split(': ')[-1]
            elif l.startswith('PrecursorMZ:'):
                if Spectrum_type == 'MS2' and l.split(': ')[-1] != 'NA':
                    try:
                        PrecursorMZ = float(l.split(': ')[-1])
                    except:
                        correct = False
                        print(DB)
                        break
            elif l.startswith('ExactMass:'):
                ExactMass = float(l.split(': ')[-1])
            elif l.startswith('Num Peaks:'):
                Num_Peaks = int(l.split(': ')[-1])
            elif ':' not in l:
                mz.append(float(l.split(' ')[0]))
                inten.append(float(l.split(' ')[1]))
        if correct:
            mz = np.array(mz)
            inten = np.array(inten)
            metadata = {'Name':Name, 'InChIKey':InChIKey,'SMILES':SMILES, 'ExactMass':ExactMass,'Num Peaks':Num_Peaks, 
                        'Spectrum_type':Spectrum_type, 'Precursor_type':Precursor_type,'PrecursorMZ':PrecursorMZ}
            spectrum = Spectrum(mz, inten, metadata)
            spectra.append(spectrum)
    if save_path:
        save_as_mgf(spectra, save_path)
    else:
        return spectra


def intmz(file_path:str, save_path:str=None):
    suppl = load_from_mgf(file_path)
    mols = [m for m in suppl]
    spectra = []
    for mol in mols:
        mz_dict = {}
        for i in range(len(mol.peaks[0])):
            mz = int(mol.peaks[0][i])
            if mz in mz_dict:
                mz_dict[mz] += mol.peaks[1][i]
            else:
                mz_dict[mz] = mol.peaks[1][i]
        mzs = []
        intens = []
        for key, value in mz_dict.items():
            mzs.append(float(key))
            intens.append(value)
        mzs = np.array(mzs)
        intens = np.array(intens)/max(intens)
        spectra.append(Spectrum(mzs, intens, mol.metadata))
    if save_path:
        save_as_mgf(spectra, save_path)
    else:
        return spectra


def sdf2mgf_MoNA(file_path:str, save_path:str=None):
    suppl = Chem.SDMolSupplier(file_path)
    mols = [mol for mol in suppl if mol]
    spectra = []
    for mol in mols:
        try:
            spectrum_type = mol.GetProp('SPECTRUM TYPE')
        except:
            continue
        if mol.GetProp('SPECTRUM TYPE') == 'MS2':
            mzs_intens = mol.GetProp('MASS SPECTRAL PEAKS').split('\n')
            mzs_intens = [(float(mz_inten.split(' ')[0]), float(mz_inten.split(' ')[1])) for mz_inten in mzs_intens]
            mzs_intens = sorted(mzs_intens, key=lambda x: x[0])
            mzs, intens = zip(*mzs_intens)
            mzs = np.array(mzs)
            intens = np.array(intens)/max(intens)
            id = mol.GetProp('ID')
            smi = Chem.MolToSmiles(mol)
            precursor_type, precursor_mz = None, None
            try:
                precursor_type = mol.GetProp('PRECURSOR TYPE')
                precursor_mz = mol.GetProp('PRECURSOR M/Z')
            except:
                pass
            exactmass = mol.GetProp('EXACT MASS')
            num_peaks = mol.GetProp('NUM PEAKS')
            spectrum = Spectrum(mzs, intens, metadata = {'ID': id, 'SMILES': smi, 'ExactMass': exactmass, 'Num Peaks': num_peaks,
                                                            'Spectrum_type': spectrum_type, 'Precursor_type': precursor_type, 'PrecursorMZ': precursor_mz})
            spectra.append(spectrum)
    if save_path:
        save_as_mgf(spectra, save_path)
    else:
        return spectra