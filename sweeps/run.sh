#!/usr/bin/env bash
# gira i 4 sweep della campagna a rotazione (round-robin), pochi trial alla
# volta ciascuno, cosi' se viene interrotto in un punto qualsiasi della run
# tutti e 4 hanno avuto piu' o meno lo stesso tempo
#
# uso: ./sweeps/run.sh <campaign.yaml> <worker_profile>
# stop: Ctrl+C (o kill sul pid stampato all'avvio)

set -uo pipefail

# read required args, bash's ${var:?msg} exits with msg if the arg is missing
campaign="${1:?uso: run.sh <campaign.yaml> <worker_profile>}"
profile="${2:?uso: run.sh <campaign.yaml> <worker_profile>}"
# uso python solo per leggere il campo campaign_name dal file yaml
campaign_name=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['campaign_name'])" "$campaign")
registry_path="sweeps/registry/${campaign_name}.json"
# how many trials each sweep gets before moving to the next one in the rotation
trials_per_round=8
retry_sleep_seconds=30

if [[ ! -f "$registry_path" ]]; then
    echo "registry non trovato: $registry_path -- lancia prima 'python -m sweeps.cli register --campaign $campaign'" >&2
    exit 1
fi

mkdir -p sweeps/logs
echo "pid di questo loop: $$ (kill $$ per fermarlo in modo pulito)"

# infinite loop: keeps rotating through every sweep in the registry until someone kills it
while true; do
    sweep_ids=$(python -c "import json; print(' '.join(json.load(open('$registry_path'))))")
    for sweep_id in $sweep_ids; do
        log_file="sweeps/logs/${sweep_id}.log"
        echo "$(date '+%F %T') avvio $trials_per_round trial su $sweep_id" | tee -a "$log_file"
        if ! python -m sweeps.cli worker --campaign "$campaign" --profile "$profile" \
                --sweep-id "$sweep_id" --count "$trials_per_round" --allow-code-mismatch \
                >> "$log_file" 2>&1; then
            # if the worker crashed just wait a bit and move to the next sweep
            echo "$(date '+%F %T') worker fallito su $sweep_id, pausa ${retry_sleep_seconds}s prima di riprovare" | tee -a "$log_file"
            sleep "$retry_sleep_seconds"
        fi
    done
done
