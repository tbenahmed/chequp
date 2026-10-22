#!/bin/bash
# 1D time-convergence study (fixed 256-cell grid, fixed dt, t = 0..10 ns).
#   h1d : full physics (hydro + ionization + implicit conduction)
#   d1d : no hydro (ionization + implicit conduction): the implicit scheme alone,
#         with dt far beyond the explicit diffusion limit
#   nod1d / hyd1d / ion1d : conduction off / hydro only / ionization only,
#         to attribute the time error
here=$(cd "$(dirname "$0")" && pwd); export WORKDIR=${WORKDIR:-$PWD/work}
mkdir -p $WORKDIR && python $here/make_ic.py $WORKDIR
G="amr.max_level=0 amr.n_cell=256 max_step=1000000"
{
for dt in 1.6e-11 8e-12 4e-12 2e-12 1e-12 5e-13; do
  echo "h1d_dt$dt $G castro.fixed_dt=$dt"
  echo "nod1d_dt$dt $G thd_diffusion.mode=off castro.fixed_dt=$dt"
  echo "hyd1d_dt$dt $G castro.add_ext_src=0 castro.fixed_dt=$dt"
  echo "ion1d_dt$dt $G castro.do_hydro=0 thd_diffusion.mode=off castro.fixed_dt=$dt"
done
for dt in 2e-10 1e-10 5e-11 2.5e-11 1.25e-11 6.25e-12 3.125e-12; do
  echo "d1d_dt$dt $G castro.do_hydro=0 castro.fixed_dt=$dt"
done
} | xargs -P ${NJOBS:-4} -L1 bash -c "$here/run_case.sh 1 \$0 1 \$@"
for s in h1d d1d nod1d hyd1d ion1d; do
  python $here/analyze_time_convergence.py $WORKDIR/runs/${s}_dt $WORKDIR/res_$s.json
done
grep -c "did not converge" $WORKDIR/runs/*/log | grep -v ":0$"
