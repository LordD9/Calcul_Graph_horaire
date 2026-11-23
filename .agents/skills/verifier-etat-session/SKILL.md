---
name: verifier-etat-session
description: Use when touching etat_session.py, empreinte.py, or how app.py keeps energy or robustness results between reruns. Runs the TLA+ model.
---

# Vérifier l'état de session (TLA+)

## Promesse vérifiée
- **PasDePerime** : un bilan énergie ou robustesse affiché a été calculé sur les entrées courantes. Sinon, l'utilisateur lit la consommation ou le retard d'une autre grille.
- **PasDAncienApresEchec** : après un recalcul raté, l'ancien résultat n'est plus affiché.

## Quand l'utiliser
Ta modification touche :
- `etat_session.py` (`lire`, `enregistrer`, `oublier`) ;
- `empreinte.py` (`empreinte`) ;
- `app.py`, `_rendre_robustesse` ou la section « Calcul énergétique » (`etat_session.lire(... "energie" ...)`) ;
- une nouvelle entrée lue par `calculer_energie_flotte` ou `evaluer_robustesse`. Elle doit alors entrer dans le tuple `entrees` passé à `etat_session`.

## Lancer
```bash
curl -sSL -o formel/tla2tools.jar https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar
formel/verifier.sh            # JAVA=... TLA_JAR=... si java n'est pas dans le PATH
```
Durée : moins de 5 s. Attendu : 4 lignes `OK`.
- `EtatSession.cfg` passe.
- `Bug_a862025.cfg`, `Bug_81bc357.cfg` et `Garde_oublier.cfg` **doivent échouer**. Chacun rejoue un vrai bug passé. S'il passe, le modèle ne voit plus ce bug : corrige le modèle, pas le cfg.

Complément e2e : `python -m pytest test_etat_session.py test_robustness_ui.py`.

## Correspondance modèle ↔ code
- `Changer` : un widget modifie une entrée ; le rerun suivant appelle `etat_session.lire`.
- `Cle(e)` : `empreinte(*entrees)`. `ChampsCle = Champs`, car toutes les entrées sont passées.
- `CalculReussi` : `etat_session.enregistrer`.
- `CalculEchoue` avec `OublierAvantCalcul = TRUE` : `etat_session.oublier` avant `evaluer_robustesse`, dans `_rendre_robustesse`.
- Hors modèle : le contenu des calculs, les widgets Streamlit eux-mêmes, et la concurrence (Streamlit exécute une session à la fois).

## Bug réel ou erreur de modèle ?
Si `EtatSession.cfg` échoue, lis la trace TLC (états numérotés).
- Une entrée a changé et `Affiche` montre l'ancien `calculeSur` : une entrée manque au tuple `entrees` dans `app.py`. C'est un vrai bug.
- La trace contredit le code (par exemple `stocke` change sans action correspondante) : le modèle est faux. Corrige-le et note-le ici.

## Couverture et synchronisation
- Fichiers couverts : `etat_session.py`, `empreinte.py`, `app.py` (`_rendre_robustesse`, section énergie).
- Dernière synchro confirmée : branche `fix/etat-session`, PR #13.
- Avance ce repère seulement après avoir relu le code couvert et constaté qu'il correspond toujours au modèle.
