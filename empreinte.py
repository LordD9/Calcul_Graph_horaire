# -*- coding: utf-8 -*-
"""Seule source des clés de cache du projet.

Une clé choisie champ par champ finit par oublier un champ : bilan énergie
réutilisé après une nouvelle grille de même taille (a862025), distance et
rampe oubliées (81bc357), id(missions) aveugle aux modifications en place.
Ici la clé couvre tous les arguments, ou le cache n'existe pas.
"""

import functools
import hashlib
import pickle


def empreinte(*objets):
    """Deux empreintes égales impliquent des objets égaux ; l'inverse n'est pas garanti."""
    return hashlib.sha256(pickle.dumps(objets, protocol=5)).hexdigest()


def memoiser(maxsize):
    def decorer(fn):
        resultats = {}

        @functools.wraps(fn)
        def enveloppe(*args, **kwargs):
            cle = empreinte(args, kwargs)
            if cle in resultats:
                return resultats[cle]
            valeur = fn(*args, **kwargs)
            if len(resultats) >= maxsize:
                resultats.clear()
            resultats[cle] = valeur
            return valeur

        enveloppe.vider = resultats.clear
        return enveloppe

    return decorer
