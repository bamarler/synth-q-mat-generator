#!/bin/bash
#SBATCH --job-name=sqmg-vasp
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=cpu          # ADJUST: DFT is CPU-heavy; pick the right partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32     # ADJUST: cores for the VASP MPI run
#SBATCH --mem=64G

# Generic SLURM template for DFT validation (structure relaxation, electronic
# structure, SOC, band structure). VASP is a licensed binary that must already
# be installed on the cluster — this script only drives it. Adjust modules,
# partition, and the VASP launch command for the professor's cluster.

set -euo pipefail

# module load vasp/6.4 intel-mpi  # ADJUST or remove depending on cluster

STRUCT_DIR="${STRUCT_DIR:?Set STRUCT_DIR to the directory holding INCAR/POSCAR/POTCAR/KPOINTS}"
cd "${STRUCT_DIR}"

echo "[hpc] node: $(hostname)  job: ${SLURM_JOB_ID:-local}  dir: ${STRUCT_DIR}"
# srun vasp_std                  # ADJUST: cluster's VASP launch command

echo "[hpc] VASP run finished"
