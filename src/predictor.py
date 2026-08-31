#!/usr/bin/env python3

import os
import sys
import joblib
import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, Fragments
from openbabel import openbabel

MODEL_DIR = "models"

if not os.path.exists(MODEL_DIR):
    print(f"Ошибка: Папка '{MODEL_DIR}' не найдена! Убедитесь, что вы сохранили туда модели.")
    sys.exit(1)

print("Загрузка ИИ-конвейера...")
try:
    variance_selector = joblib.load(os.path.join(MODEL_DIR, '1_variance_selector.pkl'))
    rfe_selector = joblib.load(os.path.join(MODEL_DIR, '2_rfe_selector.pkl'))
    model = joblib.load(os.path.join(MODEL_DIR, 'hiv_protease_model.pkl'))
    print("Все компоненты успешно загружены.")
except Exception as e:
    print(f"Ошибка при загрузке моделей: {e}")
    sys.exit(1)

def get_mol_from_file(file_path):
    """Определяет формат файла и возвращает RDKit Mol объект"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.pdb', '.pdbqt']:
        format_type = ext.replace('.', '') 
        
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats(format_type, 'smiles')

        obMol = openbabel.OBMol()
        
        if not obConversion.ReadFile(obMol, file_path):
            print(f"Ошибка OpenBabel: Не удалось прочитать файл {file_path}")
            return None

        smiles_str = obConversion.WriteString(obMol)
        smiles_output = smiles_str.split()
        smiles = smiles_output[0] if smiles_output else None
        
        return Chem.MolFromSmiles(smiles)
    
    else:
        print(f"Ошибка: Неподдерживаемый формат файла {ext}")
        return None

def get_features(mol):
    if mol is None:
        return None

    generator = rdFingerprintGenerator.GetMorganGenerator(radius= 2, fpSize= 1024)
    fp = generator.GetFingerprint(mol)
    fp_vector = [int(i) for i in fp]

    base_descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol)
    ]

    fragment_features = [
        Fragments.fr_benzene(mol),       
        Fragments.fr_Ar_OH(mol),         
        Fragments.fr_NH2(mol),           
        Fragments.fr_NH1(mol),           
        Fragments.fr_amide(mol),         
        Fragments.fr_COO(mol),           
        Fragments.fr_Ar_N(mol),          
        Fragments.fr_ester(mol),         
        Fragments.fr_ketone(mol),        
        Fragments.fr_sulfonamd(mol)      
    ]

    full_features = fp_vector + base_descriptors + fragment_features

    return np.array(full_features).reshape(1, -1)

def main():
    parser = argparse.ArgumentParser(description= 'Universal HIV-1 Protease Affinity Predictor')

    parser.add_argument('input', type= str, help= 'SMILES строка или путь к файлу (.pdb, .pdbqt)')

    args = parser.parse_args()
    input_data = args.input
    mol = None

    if os.path.isfile(input_data):
        print(f"--- Обработка файла: {input_data} ---")
        mol = get_mol_from_file(input_data)
    else:
        print(f"--- Обработка SMILES: {input_data} ---")
        mol = Chem.MolFromSmiles(input_data)

    if mol:
        features = get_features(mol)
        if features is None:
            print("Ошибка при расчете дескрипторов молекулы")
            sys.exit(1)

        try:                
            features_vt = variance_selector.transform(features)
            features_final = rfe_selector.transform(features_vt)
            predicted_energy = model.predict(features_final)[0]
            print(f"\n[РЕЗУЛЬТАТ] Предсказанная энергия: {predicted_energy:.3f} kcal/mol")

        except Exception as e:
            print(f"Ошибка при обработке признаков моделями ИИ: {e}")
            sys.exit(1)
    else:
        print("Ошибка: Не удалось распознать структуру")
        sys.exit(1)

if __name__ == "__main__":
    main()
