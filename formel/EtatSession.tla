---------------------------- MODULE EtatSession ----------------------------
(* Résultats dérivés (bilan énergie, robustesse) gardés entre deux reruns   *)
(* Streamlit. Code couvert : etat_session.py (lire, enregistrer, oublier)   *)
(* et ses appelants dans app.py (_rendre_robustesse, section énergie).      *)
(*                                                                          *)
(* Propriétés :                                                             *)
(*  PasDePerime : ce qui s'affiche a été calculé sur les entrées courantes. *)
(*  PasDAncienApresEchec : un recalcul raté n'affiche pas l'ancien résultat.*)
(*                                                                          *)
(* ChampsCle = entrées que la clé couvre. Le code passe toutes les entrées  *)
(* à empreinte(), donc ChampsCle = Champs. Les .cfg de bug rejouent les     *)
(* anciennes clés partielles (a862025, 81bc357).                            *)
EXTENDS TLC

CONSTANTS Champs, Valeurs, ChampsCle, OublierAvantCalcul

VARIABLES entrees, stocke, dernierEchec
vars == <<entrees, stocke, dernierEchec>>

Rien == {}
Cle(e) == [c \in ChampsCle |-> e[c]]

(* lire(session, nom, entrees) *)
Affiche == {r.calculeSur : r \in {r \in stocke : r.cle = Cle(entrees)}}

Init ==
    /\ entrees \in [Champs -> Valeurs]
    /\ stocke = Rien
    /\ dernierEchec = FALSE

(* Un widget change une entrée ; le rerun suivant relit le résultat. *)
Changer ==
    \E c \in Champs, v \in Valeurs :
        /\ entrees' = [entrees EXCEPT ![c] = v]
        /\ stocke' = {r \in stocke : r.cle = Cle(entrees')}
        /\ dernierEchec' = FALSE

(* enregistrer(session, nom, entrees, valeur) après un calcul réussi. *)
CalculReussi ==
    /\ stocke' = {[cle |-> Cle(entrees), calculeSur |-> entrees]}
    /\ dernierEchec' = FALSE
    /\ UNCHANGED entrees

(* Le calcul lève : seul oublier() avant calcul retire l'ancien résultat. *)
CalculEchoue ==
    /\ stocke' = IF OublierAvantCalcul THEN Rien ELSE stocke
    /\ dernierEchec' = TRUE
    /\ UNCHANGED entrees

Next == Changer \/ CalculReussi \/ CalculEchoue

Spec == Init /\ [][Next]_vars

PasDePerime == Affiche \subseteq {entrees}
PasDAncienApresEchec == dernierEchec => Affiche = Rien
=============================================================================
