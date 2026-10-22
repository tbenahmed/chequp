#!/bin/bash
# 2D rz study.
#  1. z-uniform plasma on a 256x8 grid at dt = 2e-12: must reproduce the 1D run
#     h1d_dt2e-12 of run_1d_study.sh (compare with compare_1d_2d.py).
#  2. finite plasma column, 64x64 grid, t = 0..4 ns, fixed-dt refinement.
here=$(cd "$(dirname "$0")" && pwd); export WORKDIR=${WORKDIR:-$PWD/work}
mkdir -p $WORKDIR && python $here/make_ic.py $WORKDIR
IC2D=ic_2d_uniformz.h5 $here/run_case.sh 2 uz2d_dt2e-12 1 amr.max_level=0 amr.n_cell="256 8" \
    castro.fixed_dt=2e-12 max_step=1000000 &
for dt in 1e-12 2e-12 4e-12 8e-12 1.6e-11; do echo $dt; done | xargs -P ${NJOBS:-4} -I{} \
  $here/run_case.sh 2 col2d_dt{} 1 amr.max_level=0 amr.n_cell="64 64" max_step=1000000 \
    stop_time=4e-9 amr.plot_per=1e-9 castro.fixed_dt={}
wait
python $here/analyze_time_convergence.py $WORKDIR/runs/col2d_dt $WORKDIR/res_col2d.json
python $here/compare_1d_2d.py $WORKDIR/runs/h1d_dt2e-12 $WORKDIR/runs/uz2d_dt2e-12
grep -c "did not converge" $WORKDIR/runs/*/log | grep -v ":0$"
