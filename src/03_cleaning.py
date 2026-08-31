#!/usr/bin/env python3

import os
import pandas as pd

INPUT_CSV = "data/raw/screening_results.csv"
LIGANDS_DIR = "docking/ligands"
OUTPUT_CSV = "data/processed/screening_results_cleaned.csv"

def get_rotatable_bonds(ligand_path):
    try:
        with open(ligand_path, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if line.startswith("TORSDOF"):
                    return int(line.split()[1])

            for line in lines:
                if 'active torsions' in line:
                    return int(''.join(c for c in line if c.isdigit()))

            branch_count = sum(1 for line in lines if line.startswith("BRANCH"))
            return branch_count
    except:
        return 999

def clean_csv():
    if not os.path.exists(INPUT_CSV):
        print(f"Ошибка: Не найден файл результатов докинга {INPUT_CSV}")
        return
    
    df = pd.read_csv(INPUT_CSV)
    pre_len = len(df)

    df = df[pd.to_numeric(df["Binding_Affinity_kcal_mol"], errors= 'coerce').notnull()]
    df['Binding_Affinity_kcal_mol'] = df['Binding_Affinity_kcal_mol'].astype(float)
    df = df[(df['Binding_Affinity_kcal_mol'] <= 0.0) & (df['Binding_Affinity_kcal_mol'] >= -16.0)]

    print("Фильтруем старые молекулы по количеству связей...")
    mask = df["Ligand_Name"].apply(lambda x: get_rotatable_bonds(os.path.join(LIGANDS_DIR, x)) <= 16)
    df = df[mask]

    print(f"Было молекул: {pre_len}")
    print(f"Осталось после чистки: {len(df)}")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index= False)
    print(f"Чистый датасет сохранен в {OUTPUT_CSV}")

if __name__ == "__main__":
    clean_csv()