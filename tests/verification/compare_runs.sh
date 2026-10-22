#!/bin/bash
# Compare all plotfiles (all AMR levels) of two runs with AMReX fcompare and
# print the max relative error of the main fields for each output.
#   usage: compare_runs.sh <runA> <runB>
# FCOMPARE: path to fcompare executable (build it in amrex/Tools/Plotfile with
#           `make programs=fcompare`).
FC=${FCOMPARE:-fcompare.gnu.ex}
for p in $(ls -d $1/plt* | xargs -n1 basename); do
  [ -d $2/$p ] || { echo "$p missing in $2"; continue; }
  out=$($FC -r 1e-300 $1/$p $2/$p 2>&1)
  echo "$p $(echo "$out" | awk '/^ (density|Temp|rho_e|rho_H1|rho_f_heavie|xmom) /{m[$1]=($3>m[$1]?$3:m[$1])} END{for(k in m) printf "%s=%.1e ",k,m[k]}') $(echo "$out" | grep -i -E "not match" | head -1 | sed 's/.*ERROR: //')"
done
