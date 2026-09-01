#!/usr/bin/env python3

import os
import sys
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, Fragments
import matplotlib.pyplot as plt

MODEL_DIR = "models"

if not os.path.exists(MODEL_DIR):
    print(f"Ошибка: Папка '{MODEL_DIR}' не найдена! Убедитесь, что вы сохранили туда модели.")
    sys.exit(1)

print("Загрузка ИИ-конвейера...")
variance_selector = joblib.load(os.path.join(MODEL_DIR, '1_variance_selector.pkl'))
rfe_selector = joblib.load(os.path.join(MODEL_DIR, '2_rfe_selector.pkl'))
model = joblib.load(os.path.join(MODEL_DIR, 'hiv_protease_model.pkl'))
print("Все компоненты успешно загружены.")

def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
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

    return full_features

def run_prediction(smiles, name="Unknown Molecule"):
    features = get_features(smiles)

    if features is None:
        return "Ошибка SMILES"

    try:
        features_2d = np.array(features).reshape(1, -1)
        features_vt = variance_selector.transform(features_2d)
        features_final = rfe_selector.transform(features_vt)

        predicted_energy = model.predict(features_final)[0]

        return predicted_energy
    except Exception as e:
        return f"Ошибка конвейера: {str(e)}"


test_molecules = {
    "Ritonavir (Лекарство от ВИЧ)": "CC(C)C1=NC(=CS1)CN(C)C(=O)NC(C(C)C)C(=O)NC(CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)O",
    "Aspirin (Просто контроль)": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
}

plot_drugs = []
plot_energies = []
print("=== РЕЗУЛЬТАТЫ ТЕСТА ===")
for name, smiles in test_molecules.items():
    res = run_prediction(smiles)
    if isinstance(res, str):
        print(f"{name:30} -> {res}")
    else:
        print(f"{name:30} -> Предсказанная энергия: {res:.2f} kcal/mol")
        plot_drugs.append(name)
        plot_energies.append(res)

if plot_energies:  
    print("\nГенерация графика...")
    plt.figure(figsize=(9, 6))
    
    bars = plt.bar(plot_drugs, plot_energies, color=['teal', 'gray', 'lightgray'])

    plt.bar_label(bars, fmt='%.2f', padding=-15, color='white', fontweight='bold')
    plt.ylabel('Predicted Affinity (kcal/mol)')
    plt.title('Blind Test: Specific Inhibitor vs Non-specific Compounds')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=15, ha='right')

    os.makedirs('visualization', exist_ok= True)
    plt.savefig('visualization/blind_test_results.png', dpi = 300, bbox_inches = 'tight')
    print("График сохранен в 'visualization/blind_test_results.png'")

    plt.show()

else:
    print("\nНе удалось построить график: нет успешных предсказаний.")
