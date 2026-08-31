#!/usr/bin/env python3

import os
import sys
import csv
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Fragments
from openbabel import openbabel

INPUT_CSV = "data/processed/screening_results_cleaned.csv"
LIGANDS_DIR = "docking/ligands"
OUTPUT_CSV = "data/processed/dataset_for_ml.csv"

def pdbqt_to_features(pdbqt_path):
    if not os.path.exists(pdbqt_path) or os.path.getsize(pdbqt_path) == 0:
        return None

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats('pdbqt', 'smiles')

    obMol = openbabel.OBMol()
    obConversion.ReadFile(obMol, pdbqt_path)

    smiles_output = obConversion.WriteString(obMol).split()
    smiles = smiles_output[0] if smiles_output else None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    generator = rdFingerprintGenerator.GetMorganGenerator(radius = 2, fpSize= 1024)
    fp = generator.GetFingerprint(mol)
    bit_string = ''.join([str(f) for f in fp])

    mol_wt = Descriptors.MolWt(mol)
    log_p = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)

    base_descriptors = [mol_wt, log_p, h_donors, h_acceptors, tpsa]

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

    return [bit_string] + base_descriptors + fragment_features


def built_dataset():
    if not os.path.exists(INPUT_CSV):
        print(f"Ошибка, не найден файл {INPUT_CSV}. Сначала запустите докинг-скрининг")
        sys.exit(1)

    base_headers = ["MolWeight", "LogP", "H_Donors", "H_Acceptors", "TPSA"]
    frag_headers = ["fr_benzene", "fr_Ar_OH", "fr_NH2", "fr_NH1", "fr_amide", "fr_COO", "fr_Ar_N", "fr_ester", "fr_ketone", "fr_sulfonamd"]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(INPUT_CSV, 'r') as input_file, open(OUTPUT_CSV, 'w') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        header = next(reader)
        writer.writerow(header + ["Fingerprint_1024b"] + base_headers + frag_headers)

        for row in reader:
            ligand_name = row[0]
            affinity = row[1]
            if "Error" in affinity or "Skipped" in affinity:
                continue

            ligand_path = os.path.join(LIGANDS_DIR, ligand_name)

            features = pdbqt_to_features(ligand_path)

            if features:   
                writer.writerow([ligand_name, affinity] + features)
            else:
                print(f"Предупреждение, RDKit не смог прочитать структуру {ligand_name}")
            print(f"\rОбработано молекул: {reader.line_num - 1}", end="", flush=True)
    print(f"\n База данных для обучения нейросети успешно создана: {OUTPUT_CSV}")

if __name__ == '__main__':
    built_dataset()