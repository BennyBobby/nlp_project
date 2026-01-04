#!/bin/bash

# Configuration
INPUT_FILE="book_eps_0.7_k_5.json"

echo "Lancement de l'évaluation Diversité et MAUVE..."
echo "Fichier test : $INPUT_FILE"

# Commande d'exécution
python measure_diversity_mauve_gen_length.py \
    --test_path $INPUT_FILE

echo "Évaluation Diversité et MAUVE terminée."