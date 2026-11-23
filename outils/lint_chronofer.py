# -*- coding: utf-8 -*-
"""Règles propres à Chronofer. Chaque message nomme la correction.

Usage : python outils/lint_chronofer.py [fichiers...]   (défaut : *.py du dépôt)
"""
import ast
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
SEUL_AUTORISE = {"empreinte.py"}

CORRECTION_CACHE = (
    "décore la fonction avec @empreinte.memoiser(maxsize=...) ; "
    "la clé couvre alors tous ses arguments (a862025, 81bc357)"
)


MOTS_CLE_MANUELLE = ("cache_key", "fingerprint", "empreinte")


def _cle_session_manuelle(noeud):
    if isinstance(noeud, ast.Attribute):
        nom = noeud.attr
        dans_session = isinstance(noeud.value, ast.Attribute) and noeud.value.attr == "session_state"
    else:
        nom = noeud.value if isinstance(noeud.value, str) else ""
        dans_session = False
        parent = getattr(noeud, "parent", None)
        if isinstance(parent, ast.Subscript) and isinstance(parent.value, ast.Attribute):
            dans_session = parent.value.attr == "session_state"
    return dans_session and any(mot in nom.lower() for mot in MOTS_CLE_MANUELLE)


def _lit(noeud, cle):
    return any(isinstance(c, ast.Constant) and c.value == cle for c in ast.walk(noeud))


def _fonction_englobante(noeud):
    while noeud is not None and not isinstance(noeud, ast.FunctionDef):
        noeud = getattr(noeud, "parent", None)
    return noeud.name if noeud is not None else None


def _violations(chemin, source):
    arbre = ast.parse(source, filename=str(chemin))
    for parent in ast.walk(arbre):
        for enfant in ast.iter_child_nodes(parent):
            enfant.parent = parent
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Import):
            for alias in noeud.names:
                if alias.name == "hashlib":
                    yield noeud.lineno, "CF001", f"hashlib hors d'empreinte.py : {CORRECTION_CACHE}"
        elif isinstance(noeud, ast.ImportFrom) and noeud.module == "functools":
            for alias in noeud.names:
                if alias.name in {"lru_cache", "cache"}:
                    yield noeud.lineno, "CF001", f"functools.{alias.name} : {CORRECTION_CACHE}"
        elif (isinstance(noeud, ast.Call) and isinstance(noeud.func, ast.Name)
              and noeud.func.id == "id"):
            yield noeud.lineno, "CF002", (
                "id() comme identité de données : un objet modifié en place garde "
                f"son id ; {CORRECTION_CACHE}"
            )
        elif isinstance(noeud, (ast.Attribute, ast.Constant)) and _cle_session_manuelle(noeud):
            yield noeud.lineno, "CF004", (
                "clé de cache/empreinte rangée à la main dans session_state : "
                "utilise etat_session.enregistrer/lire, qui jettent le résultat "
                "dès qu'une entrée change (a862025, formel/EtatSession.tla)"
            )
    for noeud in ast.walk(arbre):
        if (isinstance(noeud, ast.BinOp) and isinstance(noeud.op, ast.Sub)
                and _lit(noeud.left, "time_offset_min") and _lit(noeud.right, "time_offset_min")
                and _fonction_englobante(noeud) != "horaire_mission"):
            yield noeud.lineno, "CF005", (
                "arrivée moins arrivée : la durée inclut l'arrêt de la gare de départ. "
                "Écris point_suivant['time_offset_min'] - point['depart_offset_min'] "
                "(a681367, de27eba, b41ce03)"
            )
    for noeud in arbre.body:
        if isinstance(noeud, ast.Assign) and isinstance(noeud.value, ast.Dict):
            for cible in noeud.targets:
                if isinstance(cible, ast.Name) and "cache" in cible.id.lower():
                    yield noeud.lineno, "CF003", (
                        f"cache global {cible.id} écrit à la main : {CORRECTION_CACHE}"
                    )


def verifier(chemins):
    erreurs = []
    for chemin in chemins:
        if chemin.name in SEUL_AUTORISE:
            continue
        source = chemin.read_text(encoding="utf-8")
        for ligne, code, message in _violations(chemin, source):
            erreurs.append(f"{chemin.relative_to(RACINE)}:{ligne}: {code} {message}")
    return erreurs


def main(argv):
    chemins = [Path(a).resolve() for a in argv] or sorted(RACINE.glob("*.py"))
    erreurs = verifier(chemins)
    print("\n".join(erreurs) or "lint_chronofer : OK")
    return 1 if erreurs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
