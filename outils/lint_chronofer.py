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


def _violations(chemin, source):
    arbre = ast.parse(source, filename=str(chemin))
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
