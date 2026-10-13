# 🚄 Chronofer - Générateur de Graphique Horaire & Analyse Énergétique

**Chronofer** est un outil de prototypage rapide pour la conception de grilles horaires ferroviaires et l'analyse de la consommation énergétique des trains. Il permet de simuler des lignes à voie unique ou double, d'optimiser les rotations de matériel, et d'évaluer la faisabilité énergétique (notamment pour les trains à batterie).

![Logo](logo.png)

## Fonctionnalités Principales

*   **Modélisation d'Infrastructure :** Définition simple des gares, distances, types de voies et électrification.
*   **Planification des Missions :** Création de missions avec fréquences, origines, terminus et arrêts intermédiaires.
*   **Génération d'Horaires :**
    *   **Automatique (Optimisé) :** Algorithmes avancés pour minimiser le nombre de rames et optimiser les croisements.
    *   **Manuel :** Construction train par train ou **import depuis Excel**.
*   **Analyse de Performance :**
    *   **Statistiques de Flotte :** Nombre de rames, trajets par rame, kilométrage moyen.
    *   **Qualité de Service :** Analyse de la régularité du cadencement (Indice d'homogénéité / Gini).
*   **Simulation Énergétique :** Calcul précis de la consommation (Diesel, Électrique, Batterie, Bi-mode) basé sur la physique du mouvement (équation de Davis, profils de vitesse réels).

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

1.  **Infrastructure :** Saisissez la liste des gares (Format: `Nom;Position_KM;[Infra];[Electrification]`).
    *   *Infra codes :* 
        *   `F` : Voie unique standard (croisement impossible).
        *   `VE` : Voie d'évitement (croisement possible).
        *   `D` : Double voie (croisement possible sans arrêt).
        *   `T` : Terminus.
    *   *Electrification codes (Optionnel) :*
        *   `F` : Non électrifié.
        *   `C1500` : Caténaire 1500V DC.
        *   `C25` : Caténaire 25kV AC.
        *   `R<Puissance_kW>` : Point de recharge statique (ex: `R400` pour 400kW).
2.  **Missions :** Définissez vos missions (ex: Paris -> Lyon, fréquence 1 train/h).
3.  **Génération :** Sélectionnez un mode d'optimisation (obligatoire en mode automatique) pour générer le graphique.
4.  **Analyse :** Consultez le graphique espace-temps, les statistiques d'utilisation des rames et le bilan énergétique détaillé.

### 4. Modes d'Optimisation

Le système propose plusieurs algorithmes pour trouver la meilleure grille horaire :
*   **Simple :** Simulation directe respectant strictement les temps de retournement configurés.
*   **Fast :** Recherche rapide (pas de 10 min) pour les grandes instances.
*   **Smart Progressive :** Affinement intelligent et progressif du pas de temps (10min → 1min).
*   **Génétique :** Algorithme évolutionnaire parallèle explorant les décalages horaires et les stratégies de croisement.
*   **Exhaustif :** Exploration complète de l'espace (recommandé pour < 3 missions).

### 5. Mode Énergie & Batterie

Le mode Énergie simule la physique du train seconde par seconde :
*   **Résistance de Davis :** $F = A + Bv + Cv^2$.
*   **Récupération :** Freinage régénératif paramétrable.
*   **Gestion Batterie :** Suivi du SoC (State of Charge), recharge dynamique sous caténaire et statique aux gares équipées.
*   **Visualisation :** Profil de charge batterie par train.

### 6. Mode Manuel & Import Excel

En mode "Manuel", vous pouvez importer un roulement existant via un fichier Excel.
**Format attendu du fichier Excel :**
Le fichier doit contenir les colonnes suivantes :
*   `Train` : Identifiant unique du train ou de la rame.
*   `Début` : Date et heure de départ (Format datetime).
*   `Fin` : Date et heure d'arrivée (Format datetime).
*   `Origine` : Nom de la gare de départ.
*   `Terminus` : Nom de la gare d'arrivée.

### 7. Robustesse du graphique

Après génération d'un graphique, ouvrez l'onglet **Robustesse** et cliquez sur
**Calculer la robustesse**. Ce calcul indépendant ne relance pas l'optimisation.
Il teste un retard unique de **5, 10 et 20 minutes** à chaque départ et à chaque
point de croisement. Le point critique est choisi séparément pour chaque durée,
selon la somme des retards d'arrivée des missions complètes.

Le rapport compare les graphiques perturbés, le retard cumulé, le temps de retour
durable à l'horaire et les conséquences énergétiques pour les rames batterie.
« Non rétabli sur la période » signifie que le retour à la normale n'a pas pu être
constaté sur les services fournis. En cas de retard, le retournement minimal est
de **5 minutes**, sans départ anticipé.

La réserve est facultative : sélectionnez un terminus, l'autre ou les deux.
Une rame physique par gare sélectionnée peut reprendre les rotations suivantes
au départ planifié, uniquement si elle réduit le retard cumulé et respecte les
contraintes d'infrastructure, de matériel et d'énergie. Sa batterie est pleine
au sens du SoC maximal opérationnel configuré lors de sa mobilisation. Les
activations et les refus sont explicités dans le rapport.

Les résultats deviennent périmés lorsque les entrées changent. Le recalcul reste
manuel, via le même bouton.

## Auteurs & Licence

Projet développé par le Cerema.
Licence GNU General Public License v3.0 (voir fichier LICENSE).