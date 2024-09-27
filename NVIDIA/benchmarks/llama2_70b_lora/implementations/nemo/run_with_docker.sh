#!/bin/bash

set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${MODEL:?MODEL not set}"

# Vars with defaults
: "${NEXP:=1}"
: "${SEED:=${SEED-$RANDOM}}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${MLPERF_RULESET:=4.0.0}"
: "${LOGDIR:=./results}"


# Other vars
readonly _config_file="./configs/config_${DGXSYSTEM}.sh"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=llama2_lora
_cont_mounts=("--volume=${DATADIR}:/data" "--volume=${MODEL}:/ckpt" "--volume=${LOGDIR}:/results")


# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DATADIR)
_config_env+=(MODEL)
_config_env+=(DGXSYSTEM)
_config_env+=(SEED)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

cleanup_docker() {
    if docker ps --format '{{.Names}}' | grep -q "^${_cont_name}$"; then
        docker container rm -f "${_cont_name}" || true
    fi
}

cleanup_docker
trap 'set -eux; cleanup_docker' EXIT


# Setup container
docker run --gpus all --rm --init --detach \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    "${CONT}" sleep infinity
# Make sure container has time to finish initialization
sleep 5
docker exec -it "${_cont_name}" true

# Fix dependencies
docker exec -it "${_cont_name}" pip install huggingface-hub==0.23.2
docker exec -it "${_cont_name}" pip install transformers==4.40.0

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
  (
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    if [[ $CLEAR_CACHES == 1 ]]; then
      bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
    fi

    docker exec -it ${_config_env[@]} ${_cont_name} ./run_and_time.sh
  ) |& tee "${_logfile_base}_${_experiment_index}.log"

    if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      docker exec -it "${_config_env[@]}" "${_cont_name}"  \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${DATESTAMP}_${_experiment_index}.log" \
    || true
    fi
done