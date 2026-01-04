#!/bin/bash

# Configuration
MODEL_EVAL="facebook/opt-2.7b"
INPUT_FILE="book_eps_0.8_k_10.json"

echo "Lancement de l'évaluation de la cohérence..."
echo "Modèle juge : $MODEL_EVAL"
echo "Fichier test : $INPUT_FILE"

# Commande d'exécution
python compute_coherence.py \
    --opt_model_name $MODEL_EVAL \
    --test_path $INPUT_FILE

echo "Évaluation de la cohérence terminée."