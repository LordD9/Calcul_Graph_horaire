# Chronofer - Calcul de Graphique Horaire Ferroviaire

## 📋 Présentation

**Chronofer** est une application web interactive développée en Python avec **Streamlit**. Elle est conçue pour aider à la conception, l'analyse et l'optimisation de graphiques horaires ferroviaires, avec une composante forte dédiée au calcul de la consommation énergétique des trains.

L'outil permet de :
- Définir une infrastructure ferroviaire (gares, distances, voies uniques/doubles, électrification).
- Créer des missions de transport (origines, terminus, arrêts, fréquences, matériel roulant).
- Générer automatiquement des horaires optimisés (cadencés) ou les construire manuellement.
- Simuler la consommation énergétique détaillée (Diesel, Électrique, Bimode, Batterie) en tenant compte de la physique du train (équation de Davis, profils de vitesse).

## 🚀 Fonctionnalités Clés

### 1. Modélisation de l'Infrastructure
- Saisie simplifiée des gares et des distances.
- Configuration des zones de croisement (Voie d'Évitement) et des sections à voie double.
- **Nouveau :** Définition de l'électrification (Caténaire 1500V/25kV, zones de recharge) et des pentes (rampes) pour le calcul énergétique.

### 2. Gestion des Missions
- Création de missions aller/retour avec fréquences personnalisables.
- Saisie des temps de parcours et des temps de retournement.
- Gestion des points de passage intermédiaires (arrêts commerciaux ou techniques).
- Choix du type de matériel roulant par mission.

### 3. Génération d'Horaires
L'application propose plusieurs modes de calcul :
- **Mode Manuel :** Construction train par train (ajout, suppression, modification d'étapes).
- **Rotation Optimisée (Standard) :** Algorithme glouton rapide ("Smart") pour générer un graphique sans conflits.
- **Optimisation Avancée :**
    - **Smart :** Heuristique rapide.
    - **Exhaustif :** Explore toutes les combinaisons de décalage (pour petits réseaux).
    - **Génétique :** Algorithme évolutionnaire pour les cas complexes, avec **parallélisation** (multi-cœur) pour des performances accrues.
    - **Optimisation des Croisements :** Ajustement intelligent des temps d'arrêt pour résoudre les conflits sur voie unique.

### 4. Simulation Énergétique
Un module physique complet calcule la consommation pour chaque trajet :
- Prise en compte de la masse, de l'accélération, et de la résistance à l'avancement (Davis).
- Gestion des profils de vitesse (accélération, croisière, freinage).
- Simulation des batteries : état de charge (SoC), recharge dynamique sous caténaire, recharge à quai.
- Bilan détaillé : kWh consommés, litres de diesel, énergie récupérée au freinage.

### 5. Visualisation et Export
- Graphique espace-temps (tableau de marche graphique) interactif.
- Schéma de la ligne (voie unique/double).
- Graphiques d'état de charge des batteries.
- Export des données au format Excel et des graphiques en PDF.

## 🛠️ Installation et Lancement

### Prérequis
- Python 3.8 ou supérieur.
- Un environnement virtuel est recommandé (venv ou conda).

### Installation des dépendances

Installez les bibliothèques nécessaires via `pip` :

```bash
pip install -r requirements.txt
```

*(Le fichier `requirements.txt` doit contenir : streamlit, pandas, matplotlib, numpy, openpyxl, xlsxwriter)*

### Lancement de l'application

Exécutez la commande suivante dans votre terminal :

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut (généralement à l'adresse `http://localhost:8501`).

## 📂 Structure du Projet

*   **`app.py`** : Point d'entrée de l'application. Gère l'interface utilisateur Streamlit et l'orchestration des modules.
*   **`core_logic.py`** : Cœur du moteur de simulation horaire (génération des sillons, détection des conflits basiques).
*   **`optimisation_logic.py`** : Algorithmes d'optimisation avancée (Génétique, Exhaustif) et résolution de conflits par délais.
*   **`energy_logic.py`** : Moteur de calcul physique et énergétique (profils de vitesse, consommation, batterie).
*   **`plotting.py`** : Fonctions de tracé des graphiques (tableau de marche, infrastructure, batteries).
*   **`utils.py`** : Fonctions utilitaires diverses.
*   **`logo.png`** : Logo affiché dans l'application.

## ⚠️ Notes Importantes

- Le calcul d'optimisation avancée (surtout en mode génétique ou exhaustif) peut être long. Il ne se lance que lorsque vous cliquez sur le bouton **"🚀 Générer le graphique horaire"**.
- Pour le mode "Calcul Energie", assurez-vous de bien renseigner les paramètres d'infrastructure (électrification, rampes) pour des résultats pertinents.

---
*Projet développé pour le Cerema.*
