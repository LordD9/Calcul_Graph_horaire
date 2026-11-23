# -*- coding: utf-8 -*-
"""Résultats dérivés gardés entre deux reruns Streamlit.

Un résultat n'est lisible qu'en fournissant les entrées courantes : s'il a été
calculé sur d'autres entrées, il n'existe plus. Un bilan périmé ne peut donc
pas être affiché (a862025, 81bc357, 63ea5c6). Modèle : formel/EtatSession.tla.
"""

from typing import Any, NamedTuple

from empreinte import empreinte


class Resultat(NamedTuple):
    empreinte: str
    valeur: Any


def enregistrer(session, nom, entrees, valeur):
    session[nom] = Resultat(empreinte(*entrees), valeur)


def lire(session, nom, entrees):
    """(valeur, perime) ; valeur vaut None si absente ou calculée sur d'autres entrées."""
    try:
        resultat = session[nom]
    except KeyError:
        return None, False
    if resultat.empreinte == empreinte(*entrees):
        return resultat.valeur, False
    del session[nom]
    return None, True


def oublier(session, nom):
    try:
        del session[nom]
    except KeyError:
        pass
