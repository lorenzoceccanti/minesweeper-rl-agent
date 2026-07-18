#!/usr/bin/env bash
# stat_tests.sh — Esegue tutti i confronti statistici pairwise.
#
# Confronti previsti:
#   [1] Stesso algoritmo, architetture diverse  (fully-conv VS global-skip-conv)
#   [2] Stesso seed, algoritmo diverso          (dqn VS ppo)
#
# Uso: bash stat_tests.sh [--alpha 0.05]
#
# Il percorso di salvataggio dei .txt è letto da config.yaml (stats.save_path).

ALPHA=${1:-0.05}

# ---------------------------------------------------------------------------
# [1] STESSO ALGORITMO — ARCHITETTURE DIVERSE
#     fully-conv  VS  global-skip-conv
#     Fissiamo: stesso algo, stesso seed, board e mine identici.
# ---------------------------------------------------------------------------

echo "=== [1] Stesso algo, architetture diverse ==="

# DQN — seed 43
python main.py stats \
    --csv-a csv/dqn/dqn_fully-conv_9x9@10_seed43_results.csv \
    --csv-b csv/dqn/dqn_global-skip-conv_9x9@10_seed43_results.csv \
    --alpha "$ALPHA"

# DQN — seed 44
python main.py stats \
    --csv-a csv/dqn/dqn_fully-conv_9x9@10_seed44_results.csv \
    --csv-b csv/dqn/dqn_global-skip-conv_9x9@10_seed44_results.csv \
    --alpha "$ALPHA"

# DQN — seed 45
python main.py stats \
    --csv-a csv/dqn/dqn_fully-conv_9x9@10_seed45_results.csv \
    --csv-b csv/dqn/dqn_global-skip-conv_9x9@10_seed45_results.csv \
    --alpha "$ALPHA"

# PPO — seed 43
python main.py stats \
    --csv-a csv/ppo/ppo_fully-conv_9x9@10_seed43_results.csv \
    --csv-b csv/ppo/ppo_global-skip-conv_9x9@10_seed43_results.csv \
    --alpha "$ALPHA"

# PPO — seed 44
python main.py stats \
    --csv-a csv/ppo/ppo_fully-conv_9x9@10_seed44_results.csv \
    --csv-b csv/ppo/ppo_global-skip-conv_9x9@10_seed44_results.csv \
    --alpha "$ALPHA"

# PPO — seed 45
python main.py stats \
    --csv-a csv/ppo/ppo_fully-conv_9x9@10_seed45_results.csv \
    --csv-b csv/ppo/ppo_global-skip-conv_9x9@10_seed45_results.csv \
    --alpha "$ALPHA"

# ---------------------------------------------------------------------------
# [2] STESSO SEED — ALGORITMO DIVERSO
#     dqn  VS  ppo
#     Fissiamo: stessa architettura, stesso seed, board e mine identici.
# ---------------------------------------------------------------------------

echo "=== [2] Stesso seed, algoritmo diverso ==="

# fully-conv — seed 43
python main.py stats \
    --csv-a csv/dqn/dqn_fully-conv_9x9@10_seed43_results.csv \
    --csv-b csv/ppo/ppo_fully-conv_9x9@10_seed43_results.csv \
    --alpha "$ALPHA"

# fully-conv — seed 44
python main.py stats \
    --csv-a csv/dqn/dqn_fully-conv_9x9@10_seed44_results.csv \
    --csv-b csv/ppo/ppo_fully-conv_9x9@10_seed44_results.csv \
    --alpha "$ALPHA"

# fully-conv — seed 45
python main.py stats \
    --csv-a csv/dqn/dqn_fully-conv_9x9@10_seed45_results.csv \
    --csv-b csv/ppo/ppo_fully-conv_9x9@10_seed45_results.csv \
    --alpha "$ALPHA"

# global-skip-conv — seed 43
python main.py stats \
    --csv-a csv/dqn/dqn_global-skip-conv_9x9@10_seed43_results.csv \
    --csv-b csv/ppo/ppo_global-skip-conv_9x9@10_seed43_results.csv \
    --alpha "$ALPHA"

# global-skip-conv — seed 44
python main.py stats \
    --csv-a csv/dqn/dqn_global-skip-conv_9x9@10_seed44_results.csv \
    --csv-b csv/ppo/ppo_global-skip-conv_9x9@10_seed44_results.csv \
    --alpha "$ALPHA"

# global-skip-conv — seed 45
python main.py stats \
    --csv-a csv/dqn/dqn_global-skip-conv_9x9@10_seed45_results.csv \
    --csv-b csv/ppo/ppo_global-skip-conv_9x9@10_seed45_results.csv \
    --alpha "$ALPHA"