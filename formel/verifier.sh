#!/usr/bin/env bash
# Vérifie EtatSession.tla : le modèle passe, chaque bug rejoué échoue.
set -u
cd "$(dirname "$0")"
TLA_JAR="${TLA_JAR:-tla2tools.jar}"
JAVA="${JAVA:-java}"
tlc() { "$JAVA" -XX:+UseParallelGC -cp "$TLA_JAR" tlc2.TLC -noGenerateSpecTE -workers 1 -deadlock -cleanup -metadir "${TMPDIR:-/tmp}/tlc-$1" -config "$1" EtatSession.tla; }
statut=0
if tlc EtatSession.cfg > /dev/null; then echo "OK   EtatSession.cfg passe"; else echo "FAIL EtatSession.cfg : invariant violé"; tlc EtatSession.cfg | grep -A30 "violated"; statut=1; fi
for cfg in Bug_*.cfg Garde_*.cfg; do
  if tlc "$cfg" | grep -q "is violated"; then echo "OK   $cfg échoue comme attendu"; else echo "FAIL $cfg passe : le modèle ne voit plus ce bug"; statut=1; fi
done
exit $statut
