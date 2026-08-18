Ce projet implémente un classificateur SVM (Support Vector Machine) avec noyau RBF en utilisant PyTorch. Le code réalise un pipeline complet de machine 
learning : d'abord, il prétraite les données en gérant les valeurs manquantes (remplissage par la médiane pour les variables numériques et le mode pour 
les catégories), encode les variables catégorielles via one-hot encoding, puis normalise les données avec StandardScaler. Ensuite, il approxime le noyau RBF
gaussien grâce aux Random Fourier Features, une technique efficace qui transforme les données d'entrée en représentation de plus grande dimension. 
Enfin, il entraîne un modèle SVM linéaire en PyTorch sur ces features transformées en minimisant la perte hinge avec régularisation L2, optimisé 
par l'algorithme Adam. Le modèle est évalué sur un ensemble de test et retourne la précision (accuracy) obtenue. Cette approche combine la puissance
des SVM avec la flexibilité de PyTorch pour la classification binaire.
