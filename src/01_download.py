#!/usr/bin/env python3

import os
import sys
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem
from openbabel import openbabel
import pandas as pd


INPUT_CSV = "data/raw/chembl_download.csv"
OUTPUT_DIR = "docking/ligands"

def smiles_to_pdbqt_fast(csv_path, output_dir):

    max_molecules = 2000

    os.makedirs(output_dir, exist_ok= True)

    try:
        df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
    except Exception as e:
        print(f"Ошибка при чтении файла {csv_path}: {e}")
        return

    smiles_col = None
    for col in df.columns:
        if 'smiles' in col.lower():
            smiles_col = col
            break
    if not smiles_col:
        print("Ошибка: В таблице не найдена колонка со SMILES!")
        print(f"Доступные колонки в вашем файле: {list(df.columns)}")
        return

    print(f"файл ChEMBL прочитан, всего молекул в базе: {len(df)}")
    print(f"Обрабатываем первые {max_molecules} штук")

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats('pdb', 'pdbqt')

    success_count = 0

    for idx, row in df.iterrows():
        if success_count >= max_molecules:
            break

        smiles = str(row[smiles_col]).strip()

        if not smiles or smiles == 'nan':
            continue

        mol_number = success_count + 1
        mol_name = f'mol_{mol_number}'
        pdbqt_path = os.path.join(output_dir, f'{mol_name}.pdbqt')

        try:
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                continue

            mol = Chem.AddHs(mol)

            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol, params) < 0:
                if AllChem.EmbedMolecule(mol, useRandomCoords=True) < 0:
                    continue

            AllChem.MMFFOptimizeMolecule(mol)

            pdb_block = Chem.MolToPDBBlock(mol)

            obMol = openbabel.OBMol()
            obConversion.ReadString(obMol, pdb_block)

            charge_model = openbabel.OBChargeModel.FindType("Gasteiger")
            if charge_model:
                charge_model.ComputeCharges(obMol)

            pdbqt_block = obConversion.WriteString(obMol)

            with open(pdbqt_path, 'w', encoding='utf-8') as f:
                f.write(pdbqt_block)

            success_count += 1
            if success_count % 100 == 0:
                print(f"Успешно сгенерировано лигандов: {success_count} из {max_molecules}...")

        except Exception as e:
            continue

    print(f"\n[УСПЕХ] Начальный набор структур полностью подготовлен.")
    print(f"В целевой папке '{output_dir}' успешно создано {success_count} файлов .pdbqt с зарядами Gasteiger.")


if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        print(f"Ошибка! Не найден исходный файл: {INPUT_CSV}")
        print('Cкачанный файл ChEMBL должен находится в папке data/raw/ под именем chembl_download.csv')
        sys.exit(1)

    smiles_to_pdbqt_fast(INPUT_CSV, OUTPUT_DIR)
