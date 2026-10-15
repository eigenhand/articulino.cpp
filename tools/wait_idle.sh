#!/bin/bash
# Wait until the CPUs in $CPUS ("first-last", default: all) are nearly idle for 3 s in a row, so that benchmarks are not
# disturbed by other jobs on the same cores. Gives up after $MAX_WAIT seconds (default 1800) and returns 1.
CPUS=${CPUS:-0-$(( $(nproc) - 1 ))}; LO=${CPUS%-*}; HI=${CPUS#*-}; MAX_WAIT=${MAX_WAIT:-1800}; LIMIT=${LIMIT:-8}
snap() { awk -v lo=$LO -v hi=$HI '$1 ~ /^cpu[0-9]+$/ { n = substr($1, 4) + 0; if (n >= lo && n <= hi) { idle += $5 + $6; tot = 0; for (i = 2; i <= 9; i++) tot += $i; all += tot } } END { print idle, all }' /proc/stat; }
ok=0; waited=0
while [ $waited -lt $MAX_WAIT ]; do
  read i1 a1 < <(snap); sleep 1; read i2 a2 < <(snap); waited=$((waited + 1))
  busy=$(( 100 - 100 * (i2 - i1) / (a2 - a1 + 1) ))
  if [ $busy -le $LIMIT ]; then ok=$((ok + 1)); else ok=0; fi
  [ $ok -ge 3 ] && exit 0
done
exit 1
