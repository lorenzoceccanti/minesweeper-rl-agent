#!/usr/bin/env bash
# gira i 4 sweep della campagna a rotazione (round-robin), pochi trial alla
# volta ciascuno, cosi' se lo interrompi in un punto qualsiasi della notte
# tutti e 4 hanno avuto piu' o meno lo stesso tempo -- invece che il primo
# sweep che si mangia tutta la notte prima che gli altri partano
#
# uso: ./sweeps/run_overnight.sh <campaign.yaml> <worker_profile>
# stop: Ctrl+C (o kill sul pid stampato all'avvio)

set -uo pipefail

campaign="${1:?uso: run_overnight.sh <campaign.yaml> <worker_profile>}"
profile="${2:?uso: run_overnight.sh <campaign.yaml> <worker_profile>}"
campaign_name=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['campaign_name'])" "$campaign")
registry_path="sweeps/registry/${campaign_name}.json"
trials_per_round=8
retry_sleep_seconds=30

if [[ ! -f "$registry_path" ]]; then
    echo "registry non trovato: $registry_path -- lancia prima 'python -m sweeps.cli register --campaign $campaign'" >&2
    exit 1
fi

mkdir -p sweeps/logs
echo "pid di questo loop: $$ (kill $$ per fermarlo in modo pulito)"

while true; do
    sweep_ids=$(python -c "import json; print(' '.join(json.load(open('$registry_path'))))")
    for sweep_id in $sweep_ids; do
        log_file="sweeps/logs/${sweep_id}.log"
        echo "$(date '+%F %T') avvio $trials_per_round trial su $sweep_id" | tee -a "$log_file"
        if ! python -m sweeps.cli worker --campaign "$campaign" --profile "$profile" \
                --sweep-id "$sweep_id" --count "$trials_per_round" --allow-code-mismatch \
                >> "$log_file" 2>&1; then
            echo "$(date '+%F %T') worker fallito su $sweep_id, pausa ${retry_sleep_seconds}s prima di riprovare" | tee -a "$log_file"
            sleep "$retry_sleep_seconds"
        fi
    done
done
