# 🚄 Chronofer - Générateur de Graphique Horaire & Analyse Énergétique

**Chronofer** est un outil de prototypage rapide pour la conception de grilles horaires ferroviaires et l'analyse de la consommation énergétique des trains. Il permet de simuler des lignes à voie unique ou double, d'optimiser les rotations de matériel, et d'évaluer la faisabilité énergétique (notamment pour les trains à batterie).

![Logo](logo.png)

## Fonctionnalités Principales

*   **Modélisation d'Infrastructure :** Définition simple des gares, distances, et types de voies (voie unique, évitement, double voie).
*   **Planification des Missions :** Création de missions avec fréquences, origines, terminus et arrêts intermédiaires.
*   **Génération d'Horaires :**
    *   **Automatique (Optimisé) :** Algorithme génétique (parallélisé) pour minimiser le nombre de rames et optimiser les croisements.
    *   **Manuel :** Construction train par train ou **import depuis Excel**.
*   **Analyse de Performance :**
    *   **Statistiques de Flotte :** Nombre de rames, trajets par rame, kilométrage moyen.
    *   **Qualité de Service :** Analyse de la régularité du cadencement (Indice d'homogénéité).
*   **Simulation Énergétique :** Calcul précis de la consommation (Diesel, Électrique, Batterie) basé sur la physique du mouvement (équation de Davis).

## Guide d'Utilisation Rapide

### 1. Installation

Assurez-vous d'avoir Python installé. Installez les dépendances :

```bash
pip install -r requirements.txt
```

### 2. Lancement

Lancez l'application Streamlit :

```bash
streamlit run app.py
```

### 3. Workflow Typique

1.  **Infrastructure :** Saisissez la liste des gares (Format: `Nom;Position_KM;[Infra]`).
    *   *Infra codes :* `F` (Voie unique standard), `VE` (Voie d'évitement/Croisement possible), `D` (Double voie), `T` (Terminus).
2.  **Missions :** Définissez vos missions (ex: Paris -> Lyon, fréquence 1 train/h).
3.  **Génération :**
    *   Cliquez sur "Générer le graphique horaire" pour laisser l'algorithme optimiser les croisements.
    *   Ou passez en mode "Manuel" pour importer un fichier Excel existant.
4.  **Analyse :** Consultez le graphique espace-temps, les statistiques d'utilisation des rames et, si activé, le bilan énergétique.

### 4. Mode Manuel & Import Excel

En mode "Manuel", vous pouvez importer un roulement existant via un fichier Excel.
**Format attendu du fichier Excel :**
Le fichier doit contenir les colonnes suivantes :
*   `Train` : Identifiant unique du train ou de la rame.
*   `Début` : Date et heure de départ (Format datetime).
*   `Fin` : Date et heure d'arrivée (Format datetime).
*   `Origine` : Nom de la gare de départ.
*   `Terminus` : Nom de la gare d'arrivée.

### 5. Indicateurs de Performance

*   **Rames utilisées :** Nombre total de rames nécessaires pour assurer le service.
*   **Km moyen / rame :** Indicateur d'efficience de l'utilisation du matériel roulant.
*   **Indice d'homogénéité :** Mesure la régularité des intervalles entre les trains (1.0 = cadencement parfait).

## Auteurs & Licence

Projet développé par le Cerema.
Licence GNU General Public License v3.0 (voir fichier LICENSE).