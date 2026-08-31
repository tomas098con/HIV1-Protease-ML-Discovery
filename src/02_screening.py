#!/usr/bin/env python3

import os
import sys
import subprocess
import csv
import multiprocessing as mp


MAX_ROTATABLE = 16
RECEPTOR = 'docking/receptor.pdbqt'
CONFIG = 'docking/config.txt'
LIGANDS_DIR = 'docking/ligands'
OUTPUT_DIR = 'docking/docking_results'
OUTPUT_CSV = 'data/raw/screening_results.csv'


def get_rotatable_bonds(pdbqt_path):
    count = 0
    try:
        with open(pdbqt_path, 'r') as f:
            for line in f:
                if line.startswith('BRANCH'):
                    count += 1
        return count
    except:
        return 999

def get_best_affinity(pdbqt_output):
    if os.path.exists(pdbqt_output) and os.path.getsize(pdbqt_output) > 0:
        with open(pdbqt_output, 'r') as f:
            for line in f:
                if 'REMARK VINA RESULT:' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        return float(parts[3])
    return None

def dock_molecule(ligand_name):
    ligand_path = os.path.join(LIGANDS_DIR, ligand_name)
    output_path = os.path.join(OUTPUT_DIR, f'docking_{ligand_name}')

    existing_energy = get_best_affinity(output_path)
    if existing_energy is not None:
        return ligand_name, existing_energy

    n_rot = get_rotatable_bonds(ligand_path)
    if n_rot > MAX_ROTATABLE:
        return (ligand_name, f"Skipped ({n_rot} bonds)")

    commands = [
    'vina',
    '--receptor', RECEPTOR,
    '--config', CONFIG,
    '--ligand', ligand_path,
    '--out', output_path,
    '--cpu', '1'
    ]

    try:
        subprocess.run(commands, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        energy = get_best_affinity(output_path)
        return (ligand_name, energy if energy is not None else "Error (No energy text)")
    except Exception as e:
        return (ligand_name, "Error (Vina crashed)")

def run_screening():
    if not all(os.path.exists(f) for f in [RECEPTOR, CONFIG, LIGANDS_DIR]):
        print("Ошибка: Запуск скрипта производится в другой папке")
        print(f"Проверьте наличие:\n - {RECEPTOR}\n - {CONFIG}\n - {LIGANDS_DIR}")
        sys.exit(1)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ligand_files = [f for f in os.listdir(LIGANDS_DIR) if f.endswith('.pdbqt')]

    results = []

    cores = os.cpu_count() or 1
    print(f"Запуск докинга на {cores} параллельных процессах...")
    print(f"Лимит вращающихся связей: {MAX_ROTATABLE}")
    print(f"Всего молекул в очереди: {len(ligand_files)}")
    with mp.Pool(processes= cores) as pool:
        for i, res in enumerate(pool.imap(dock_molecule, ligand_files), 1):
            results.append(res)
            sys.stdout.write(f"\rПрогресс: {i}/{len(ligand_files)} | Текущая молекула: {res[0]} -> {res[1]} kcal/mol    ")
            sys.stdout.flush()

    print()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Ligand_Name", "Binding_Affinity_kcal_mol"])
        for name, energy in results:
            writer.writerow([name, energy])

    print(f"\n\n Конвейер полностью завершен, все результаты собраны в: {OUTPUT_CSV}")

if __name__ == '__main__':
    run_screening()
