#!/bin/bash
# Reproducibility of a production-like 1D run (AMR max_level=2, adaptive CFL dt):
#   same run twice, solver cache on/off, 1/2/4 MPI ranks, tighter solver tolerances.
# Needs a working MPI launcher (MPIRUN) for the multi-rank cases and FCOMPARE.
here=$(cd "$(dirname "$0")" && pwd); export WORKDIR=${WORKDIR:-$PWD/work}
mkdir -p $WORKDIR && python $here/make_ic.py $WORKDIR
A="amr.n_cell=64 amr.max_level=2 amr.max_grid_size=16 amr.blocking_factor=8"
$here/run_case.sh 1 rep_np1  1 $A
$here/run_case.sh 1 rep_np1b 1 $A
$here/run_case.sh 1 rep_nocache 1 $A thd_diffusion.use_cache=0
$here/run_case.sh 1 rep_tight 1 $A thd_diffusion.newton_rtol=1e-10 thd_diffusion.gmres_rtol=1e-8
$here/run_case.sh 1 rep_np2  2 $A
$here/run_case.sh 1 rep_np4  4 $A
$here/run_case.sh 1 rep_np4b 4 $A
grep -H "MPI initialized with [0-9]* MPI" $WORKDIR/runs/rep_*/log
R=$WORKDIR/runs
for c in "np1 np1b" "np1 nocache" "np4 np4b" "np1 np2" "np1 np4" "np1 tight"; do
  set -- $c; echo "=== rep_$1 vs rep_$2"; $here/compare_runs.sh $R/rep_$1 $R/rep_$2
done
