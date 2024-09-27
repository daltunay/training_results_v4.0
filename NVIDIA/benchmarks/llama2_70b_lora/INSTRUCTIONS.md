## LLaMA2-70B LoRA Training Guide

### Initial Setup

1. **Setup Constants**
    ```bash
    BASE_PATH="/persistent_storage/daniel/"
    BENCHMARK_PATH="${BASE_PATH}/training_results_v4.0/NVIDIA/benchmarks/llama2_70b_lora"
    NEMO_PATH="${BENCHMARK_PATH}/implementations/nemo"
    DOCKER_IMAGE_NAME="mlperf-nvidia:lora-pytorch"
    CONTAINER_NAME="mlperf-container"
    ```

2. **Clone Repository and Create Directories**
    ```bash
    cd $BASE_PATH
    git clone https://github.com/mlcommons/training_results_v4.0.git
    cd $BENCHMARK_PATH
    mkdir -p resources/dataset resources/model
    ```

### Code Modifications
1. **Update `scripts/convert_model.py`**
    - Change `param_to_weights` from `float` to `bfloat16` to avoid crash at layer 66
    ```diff
    @@ -174,7 +174,7 @@ def convert(args):
            "transformer_engine", False
        ), "mcore_gpt transformer_engine must be enabled (or disabled) together."
    
    -    param_to_weights = lambda param: param.float()
    +    param_to_weights = lambda param: param.bfloat16()
    
        checkpoint = OrderedDict()
        checkpoint["state_dict"] = OrderedDict()
    ```

2. **Move Config Files**
    ```bash
    cd $NEMO_PATH
    mv config_*.sh configs/
    ```

3. **Update Run Script: `run_and_time.sh`**
    - Change `nproc_per_node` from `8` to `2` (number of GPUs)
    ```diff
    @@ -55,7 +55,7 @@ if [[ -n "${SLURM_LOCALID-}" ]] && [[ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NOD
        CMD=( 'bindpcie' ${CPU_EXCLUSIVE} ${IB_BIND} '--' ${NSYSCMD} 'python' '-u')
    else
        # interactive run on single node, no need to bind
    -    CMD=( ${NSYSCMD} 'torchrun' '--nproc_per_node=8' )
    +    CMD=( ${NSYSCMD} 'torchrun' '--nproc_per_node=2' )
    fi
    
    if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]
    ```

4. **Update megatron config: `megatron_gpt_peft_lora_tuning_config.yaml`**
    - Change `DGXNGPU` from `8` to `2` (number of GPUs)
    ```diff
    @@ -4,7 +4,7 @@ defaults:
    name: megatron_gpt_peft_lora_tuning
    
    trainer:
    -  devices: ${oc.decode:${oc.env:DGXNGPU,8}}
    +  devices: ${oc.decode:${oc.env:DGXNGPU,2}}
    num_nodes: ${oc.decode:${oc.env:DGXNNODES,1}}
    accelerator: gpu
    precision: ${oc.decode:${oc.env:PRECISION,bf16-mixed}}
    ```

5. **Create custom config: `configs/config_flex.sh`**
    - This config lowers parallelism and sets number of GPUs to 2.
    ```bash
    cat << 'EOF' > $NEMO_PATH/configs/config_flex.sh
    #!/bin/bash

    source $(dirname ${BASH_SOURCE[0]})/config_common.sh

    # hyperparameters
    export MAX_STEPS=896
    export LR=0.0005
    export MINIBS=4

    export TP=1
    export PP=1
    export CP=1
    export SP=1
    export TP_COMM_OVERLAP=True
    export VBOOST_VALUE=1

    export FP8=True
    export FP8_AMAX_ALGO=max
    export FP8_REDUCE_AMAX=False
    export FP8_AMAX_HISTORY=32

    export SKIP_EVALS=3
    export HYDRA_FULL_ERROR=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1

    # system parameters
    export DGXNNODES=1
    export DGXNGPU=2  # override common config
    export WALLTIME_MINUTES=45
    export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))
    EOF
    ```

### Docker Configuration
1. **(Optional) Stop and Remove Existing Container**
    ```bash
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    ```

2. **Build Docker Image**
    ```bash
    docker build -t $DOCKER_IMAGE_NAME $NEMO_PATH
    ```

3. **Run Docker Container**
    ```bash
    docker run \
        -it \
        --gpus all \
        --name $CONTAINER_NAME \
        --volume $BENCHMARK_PATH/resources/dataset:/data \
        --volume $BENCHMARK_PATH/resources/model:/model \
        --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        $DOCKER_IMAGE_NAME \
        bash
    ```

4. **Install Specific Package Versions**
    ```bash
    pip install huggingface-hub==0.23.2  # fix for https://github.com/NVIDIA/NeMo/issues/9793
    pip install transformers==4.40.0  # fix for https://github.com/NVIDIA/NeMo/issues/9272
    ```

### Dataset Preparation

Inside of the Docker container:

1. **Download Dataset**
    ```bash
    python scripts/download_dataset.py  # causes wrong hash error (non-breaking)
    ```

2. **Convert Dataset**
    ```bash
    python scripts/convert_dataset.py  # causes wrong hash error (non-breaking)
    ```

### Model Preparation

Inside of the Docker container:

1. **Download Model**
    ```bash
    python scripts/download_model.py  # causes wrong hash error (non-breaking)
    ```

2. **Convert Model**
    ```bash
    python scripts/convert_model.py \
        --input_name_or_path /model \
        --output_path /model/llama2-70b.nemo \
        --hparams_file scripts/megatron_llama_config.yaml \
        --precision bf16
    ```

3. **Extract Model**
    ```bash
    cd /model
    tar -xvf llama2-70b.nemo
    ```

4. **(Optional) Remove Other Files**
    ```bash
    find . -type f ! -name 'llama2-70b.nemo' -exec rm -f {} +
    ```

### Training Execution

1. **Create `nemo/run_with_docker.sh` Script**
    - This script was adapted from `smc/benchmarks/llama2-70b_lora/implementations/nemo/run_with_docker.sh`
    ```bash
    cat << 'EOF' > $NEMO_PATH/run_with_docker.sh
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
    EOF
    ```

2. **Start Training**
    ```bash
    cd $NEMO_PATH
    source configs/config_flex.sh
    
    CONT=$DOCKER_IMAGE_NAME \
    LOGDIR="$BENCHMARK_PATH/output" \
    DATADIR="$BENCHMARK_PATH/resources/dataset" \
    MODEL="$BENCHMARK_PATH/resources/model" \
    ./run_with_docker.sh
    ```

---

Current issue :
```text
[smc-h100x8-bm0-vm3-2accels:221  :0:221] Caught signal 11 (Segmentation fault: address not mapped to object at address 0x28)
```