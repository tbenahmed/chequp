#!/bin/bash
# Run one Castro case of the verification study.
#   usage: run_case.sh <dim> <name> <nproc> [extra Castro runtime args...]
# Environment:
#   WORKDIR  where runs/<name> and the initial conditions live (default: ./work)
#   EXEDIR   directory containing Castro{1,2}d*.gamma_law_2T.ex (default: sim_folder/build)
#   MPIRUN   MPI launcher (default: mpirun). Check that "MPI initialized with N
#            MPI processes" in the log matches N: a launcher/library mismatch
#            silently starts N independent serial runs.
#   IC2D     2D initial condition file name (default: ic_2d_column.h5)
here=$(cd "$(dirname "$0")" && pwd)
dim=$1; name=$2; np=$3; shift 3
B=${EXEDIR:-$here/../../sim_folder/build}; R=$here/../../sim_folder/run
W=${WORKDIR:-$PWD/work}
rm -rf "$W/runs/$name"; mkdir -p "$W/runs/$name"; cd "$W/runs/$name" || exit 1
if [ "$dim" = 1 ]; then exe=$(ls $B/Castro1d*.gamma_law_2T.ex | head -1); inp=$R/inputs.1d.cyl; ic=$W/ic_1d.h5
else exe=$(ls $B/Castro2d*.gamma_law_2T.ex | head -1); inp=$R/inputs.2d.cyl; ic=$W/${IC2D:-ic_2d_column.h5}; fi
# implicit JFNK conduction on, Castro explicit conduction off; outputs at exact times
common="castro.diffuse_temp=0 thd_diffusion.mode=implicit thd_diffusion.verbose=1
        amr.plot_int=-1 amr.plot_per=2e-9 castro.plot_per_is_exact=1 castro.sum_interval=-1
        amr.checkpoint_files_output=0 problem.initial_conditions_file=$ic"
start=$(date +%s.%N)
${MPIRUN:-mpirun} -np $np $exe $inp $common "$@" > log 2>&1
echo "exit=$? wall=$(echo "$(date +%s.%N) - $start" | bc)" >> log
