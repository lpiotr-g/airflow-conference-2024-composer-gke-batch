import json
from dataclasses import dataclass
from typing import Optional

from airflow import models
from airflow.providers.google.cloud.operators.kubernetes_engine import (
    GKECreateClusterOperator,
    GKEDeleteClusterOperator,
    GKEStartKueueInsideClusterOperator,
    GKECreateCustomResourceOperator,
    GKEStartKueueJobOperator,
    GKEStartJobOperator,
)
from airflow.utils.dates import days_ago

from kubernetes.client import (
    V1ResourceRequirements,
    V1Volume,
    V1VolumeMount,
    V1CSIVolumeSource,
)

from airflow.providers.google.cloud.operators.vertex_ai.model_service import (
    UploadModelOperator,
)
from airflow.providers.google.cloud.operators.vertex_ai.endpoint_service import (
    CreateEndpointOperator,
    DeployModelOperator,
)


# GLOBAL CONSTANTS
# Credentials required to run the DAG
KAGGLE_USERNAME = ""
KAGGLE_KEY = ""

# GCP project config
GCP_PROJECT = ""
GCP_REGION = ""
GCP_CLUSTER_NAME = ""

# GCS related constants
GCS_BUCKET_NAME = "example-model-trainings"
GCS_VOLUME_MOUNT = [
    V1VolumeMount(
        name="gcs-fuse-csi",
        mount_path="/data",
    ),
]

GCS_VOLUME = [
    V1Volume(
        name="gcs-fuse-csi",
        csi=V1CSIVolumeSource(
            driver="gcsfuse.csi.storage.gke.io",
            volume_attributes={
                "bucketName": GCS_BUCKET_NAME,
                "options": "--implicit-dirs",
            },
        ),
    ),
]

GCS_ANNOTATION = {
    "gke-gcsfuse/volumes": "true",
}

# Fine-tuning constants
GCS_FT_DATASET_PATH = "/data/databricks-dolly-15k.jsonl"
GCS_FT_DATASET_BIS_PATH = "/data/databricks-dolly-15k-BIS.jsonl"
DATABRICKS_DOLLY_15K_JSONL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"

# Images
PREPROCESSING_APP_LATEST = "us-central1-docker.pkg.dev/lpiotr-composer-prod-1-411312/polish-punk-depot/llm-preprocessing-app:latest"
LLM_CONVERT_APP_LATEST = "us-central1-docker.pkg.dev/lpiotr-composer-prod-1-411312/polish-punk-depot/llm-convert-app:latest"
FINETUNING_APP_LATEST = "us-central1-docker.pkg.dev/lpiotr-composer-prod-1-411312/polish-punk-depot/llm-finetuning-app:latest"
VLLM_DOCKER_URI = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240220_0936_RC01"
BENCHMARK_APP_LATEST = "us-central1-docker.pkg.dev/lpiotr-composer-prod-1-411312/polish-punk-depot/llm-benchmark-app:latest"

# Task templates
fine_tuned_path_template = "/data/fine-tuned{id_}.jsonl"
fine_tuned_path_bis_template = "/data/fine-tuned-bis{id_}.jsonl"
fine_tuned_model_path_template = "/data/gemma-finetuned{id_}"
fine_tuned_model_path_bis_template = "/data/gemma-finetuned-bis{id_}"
converted_model_path_template = "/data/gemma-finetuned-inference{id_}"


@dataclass
class ProjectConfig:
    """Class for keeping track of an item in inventory."""

    project_id: str
    cluster_region: str
    cluster_name: str


@dataclass
class JobConfig:
    """Class for keeping track of an item in inventory."""

    job_id: int
    local_queue: str


def build_resource_requirements(
    cpu_request: str,
    memory_request: str,
    cpu_limit: Optional[str] = None,
    memory_limit: Optional[str] = None,
):
    """
    Builds Kubernetes resource requirements.
    """
    return V1ResourceRequirements(
        requests={
            "cpu": cpu_request,
            "memory": memory_request,
        },
        limits={
            "cpu": cpu_limit or cpu_request,
            "memory": memory_limit or memory_request,
        },
    )


def build_kueue_preprocess_job(
    project_config: ProjectConfig,
    items: int,
    job_id: int,
    task_id: str = "kueue_job_preprocess",
    local_queue: str = "local-queue",
    input_dataset: str = GCS_FT_DATASET_PATH,
    output_file: str = fine_tuned_path_template,
):
    """
    Builds a Kueue job for preprocessing data.
    """
    return GKEStartKueueJobOperator(
        task_id=f"{task_id}{job_id}",
        project_id=project_config.project_id,
        location=project_config.cluster_region,
        cluster_name=project_config.cluster_name,
        queue_name=local_queue,
        namespace="default",
        parallelism=1,
        image=PREPROCESSING_APP_LATEST,
        name="llm-data-preprocessing",
        suspend=True,
        container_resources=build_resource_requirements("200m", "200Mi"),
        env_vars={
            "INPUT_FILE": input_dataset,
            "OUTPUT_FILE": output_file.format(id_=job_id),
            "NUM_ITEMS": str(items),
        },
        service_account_name="default",
        volume_mounts=GCS_VOLUME_MOUNT,
        volumes=GCS_VOLUME,
        annotations=GCS_ANNOTATION,
        wait_until_job_complete=True,
    )


def build_kueue_finetuning_job(
    project_config: ProjectConfig,
    steps: int,
    job_id: int,
    task_id: str = "kueue_job_finetune",
    local_queue: str = "local-queue",
    input_ft_data_path: str = fine_tuned_path_template,
    output_model_path: str = fine_tuned_model_path_template,
):
    """
    Builds a Kueue job for fine-tuning data.
    """
    return GKEStartKueueJobOperator(
        task_id=f"{task_id}{job_id}",
        project_id=project_config.project_id,
        location=project_config.cluster_region,
        cluster_name=project_config.cluster_name,
        queue_name=local_queue,
        namespace="default",
        parallelism=1,
        image=FINETUNING_APP_LATEST,
        name="llm-data-finetuning",
        suspend=True,
        container_resources=build_resource_requirements(
            "7000m", "20Gi", memory_limit="25Gi"
        ),
        env_vars={
            "INPUT_FILE": input_ft_data_path.format(id_=job_id),
            "OUTPUT_DIRECTORY": output_model_path.format(id_=job_id),
            "STEPS": str(steps),
            "KAGGLE_USERNAME": KAGGLE_USERNAME,
            "KAGGLE_KEY": KAGGLE_KEY,
        },
        service_account_name="default",
        volume_mounts=GCS_VOLUME_MOUNT,
        volumes=GCS_VOLUME,
        annotations=GCS_ANNOTATION,
        wait_until_job_complete=True,
    )


def build_kueue_convert_job(project_config: ProjectConfig, job_id: int):
    """
    Builds a Kueue job for converting models from Keras NLP to VLLM.
    """
    return GKEStartKueueJobOperator(
        task_id=f"kueue_job_convert{job_id}",
        project_id=project_config.project_id,
        location=project_config.cluster_region,
        cluster_name=project_config.cluster_name,
        queue_name="local-queue",
        namespace="default",
        parallelism=1,
        image=LLM_CONVERT_APP_LATEST,
        name="llm-conversion",
        suspend=True,
        container_resources=build_resource_requirements("6000m", "50Gi"),
        env_vars={
            "MODEL_WEIGHT_FILE": f"{fine_tuned_model_path_template.format(id_=job_id)}/model.weights.h5",
            "MODEL_VOCAB_FILE": f"{fine_tuned_model_path_template.format(id_=job_id)}/vocabulary.spm",
            "OUTPUT_DIR": converted_model_path_template.format(id_=job_id),
            "KAGGLE_USERNAME": KAGGLE_USERNAME,
            "KAGGLE_KEY": KAGGLE_KEY,
        },
        service_account_name="default",
        volume_mounts=GCS_VOLUME_MOUNT,
        volumes=GCS_VOLUME,
        annotations=GCS_ANNOTATION,
        wait_until_job_complete=True,
    )


with models.DAG(
    "kueue_finetuning_two_teams",
    schedule_interval=None,
    start_date=days_ago(1),
    tags=["example"],
) as dag:
    # DAG's CONSTANTS
    PROJECT_CONFIG = ProjectConfig(GCP_PROJECT, GCP_REGION, GCP_CLUSTER_NAME)
    WORKLOAD_IDENTITY_POOL = GCP_PROJECT + ".svc.id.goog"
    CLUSTER = {
        "name": GCP_CLUSTER_NAME,
        "workload_identity_config": {"workload_pool": WORKLOAD_IDENTITY_POOL},
        "addons_config": {"gcs_fuse_csi_driver_config": {"enabled": True}},
        "node_pools": [
            {
                "name": "autoscaled-pool-tiny",
                "initialNodeCount": 1,
                "autoscaling": {
                    "enabled": True,
                    "minNodeCount": 1,
                    "maxNodeCount": 3,
                },
                "config": {
                    "labels": {
                        "workflow-type": "tiny",
                    },
                    "workload_metadata_config": {"mode": "GKE_METADATA"},
                },
            },
            {
                "name": "autoscaled-pool-highmem",
                "initialNodeCount": 1,
                "autoscaling": {
                    "enabled": True,
                    "minNodeCount": 1,
                    "maxNodeCount": 1,
                },
                "config": {
                    "machine_type": "n4-standard-16",
                    "labels": {
                        "workflow-type": "high-mem",
                    },
                    "workload_metadata_config": {"mode": "GKE_METADATA"},
                },
                "locations": ["us-central1-a"],
            },
            {
                "name": "autoscaled-pool-mediummem",
                "initialNodeCount": 0,
                "autoscaling": {
                    "enabled": True,
                    "minNodeCount": 0,
                    "maxNodeCount": 2,
                },
                "config": {
                    "machine_type": "n4-standard-8",  # Specify the machine type
                    "labels": {
                        "workflow-type": "high-mem",
                    },
                    "workload_metadata_config": {"mode": "GKE_METADATA"},
                },
                "locations": ["us-central1-a"],  # Specify the zone for this node pool
            },
        ],
    }
    VLLM_ARGS = [
        "--host=0.0.0.0",
        "--port=7080",
        "--tensor-parallel-size=1",
        "--swap-space=16",
        "--gpu-memory-utilization=0.95",
        "--max-model-len=1",
        "--dtype=bfloat16",
        "--disable-log-stats",
    ]
    FT_TASK1_ITEMS = 1
    FT_TASK2_ITEMS = 3
    FT_TASK3_ITEMS = 5
    FT_TASK4_ITEMS = 10
    FT_TASKS_ITEMS = [FT_TASK1_ITEMS, FT_TASK2_ITEMS, FT_TASK3_ITEMS, FT_TASK4_ITEMS]
    FT_TASK1_BIS_ITEMS = 5
    FT_TASK2_BIS_ITEMS = 10
    FT_TASKS_BIS_ITEMS = [FT_TASK1_BIS_ITEMS, FT_TASK2_BIS_ITEMS]
    BENCHMARK_STEPS = 5

    tiny_flavor_conf = """
    apiVersion: kueue.x-k8s.io/v1beta1
    kind: ResourceFlavor
    metadata:
      name: tiny-cpu
    spec:
      nodeLabels:
        workflow-type: tiny
    """

    highmem_flavor_conf = """     
    apiVersion: kueue.x-k8s.io/v1beta1
    kind: ResourceFlavor
    metadata:
      name: high-mem
    spec:
      nodeLabels:
        workflow-type: high-mem
    """

    local_conf = """
    apiVersion: kueue.x-k8s.io/v1beta1
    kind: LocalQueue
    metadata:
        name: local-queue
    spec:
        clusterQueue: cluster-queue 
    """

    local_conf_bis = """
    apiVersion: kueue.x-k8s.io/v1beta1
    kind: LocalQueue
    metadata:
        name: local-queue-bis
    spec:
        clusterQueue: cluster-queue 
    """

    cluster_conf = """
    apiVersion: kueue.x-k8s.io/v1beta1
    kind: ClusterQueue
    metadata:
      name: "cluster-queue"
    spec:
      namespaceSelector: {} # match all.
      resourceGroups:
      - coveredResources: ["cpu", "memory", "pods"]
        flavors:
        - name: "tiny-cpu"
          resources:
          - name: "cpu"
            nominalQuota: 3
          - name: "memory"
            nominalQuota: 9Gi
          - name: "pods"
            nominalQuota: 30
        - name: "high-mem"
          resources:
          - name: "cpu"
            nominalQuota: 29
          - name: "memory"
            nominalQuota: 124Gi
          - name: "pods"
            nominalQuota: 4
    """

    # GKE setup tasks
    create_cluster = GKECreateClusterOperator(
        task_id="create_cluster",
        project_id=PROJECT_CONFIG.project_id,
        location=PROJECT_CONFIG.cluster_region,
        body=CLUSTER,
    )

    add_kueue_cluster = GKEStartKueueInsideClusterOperator(
        task_id="add_kueue_cluster",
        project_id=PROJECT_CONFIG.project_id,
        location=PROJECT_CONFIG.cluster_region,
        cluster_name=PROJECT_CONFIG.cluster_name,
        kueue_version="v0.8.0",
    )

    create_tiny_resource_flavor = GKECreateCustomResourceOperator(
        task_id="create_tiny_resource_flavor",
        project_id=PROJECT_CONFIG.project_id,
        location=PROJECT_CONFIG.cluster_region,
        cluster_name=PROJECT_CONFIG.cluster_name,
        yaml_conf=tiny_flavor_conf,
        custom_resource_definition=True,
        namespaced=False,
    )

    create_highmem_resource_flavor = GKECreateCustomResourceOperator(
        task_id="create_highmem_resource_flavor",
        project_id=PROJECT_CONFIG.project_id,
        location=PROJECT_CONFIG.cluster_region,
        cluster_name=PROJECT_CONFIG.cluster_name,
        yaml_conf=highmem_flavor_conf,
        custom_resource_definition=True,
        namespaced=False,
    )

    create_cluster_queue = GKECreateCustomResourceOperator(
        task_id="create_cluster_queue",
        project_id=PROJECT_CONFIG.project_id,
        location=PROJECT_CONFIG.cluster_region,
        cluster_name=PROJECT_CONFIG.cluster_name,
        yaml_conf=cluster_conf,
        custom_resource_definition=True,
        namespaced=False,
    )

    create_local_queue = GKECreateCustomResourceOperator(
        task_id="create_local_queue",
        project_id=PROJECT_CONFIG.project_id,
        location=PROJECT_CONFIG.cluster_region,
        cluster_name=PROJECT_CONFIG.cluster_name,
        yaml_conf=local_conf,
        custom_resource_definition=True,
    )

    create_local_queue_bis = GKECreateCustomResourceOperator(
        task_id="create_local_queue_bis",
        project_id=PROJECT_CONFIG.project_id,
        location=PROJECT_CONFIG.cluster_region,
        cluster_name=PROJECT_CONFIG.cluster_name,
        yaml_conf=local_conf_bis,
        custom_resource_definition=True,
    )

    job_fetch_data = GKEStartJobOperator(
        task_id="job_fetch_llm_raw_data",
        project_id=PROJECT_CONFIG.project_id,
        location=PROJECT_CONFIG.cluster_region,
        cluster_name=PROJECT_CONFIG.cluster_name,
        namespace="default",
        parallelism=1,
        image="google/cloud-sdk:latest",
        name="fetch-llm-raw-data",
        cmds=[
            "curl",
            "-L",
            "-o",
            GCS_FT_DATASET_PATH,
            DATABRICKS_DOLLY_15K_JSONL,
        ],
        container_resources=build_resource_requirements("200m", "200Mi"),
        service_account_name="default",
        volume_mounts=GCS_VOLUME_MOUNT,
        volumes=GCS_VOLUME,
        annotations=GCS_ANNOTATION,
        wait_until_job_complete=True,
    )

    job_fetch_data_bis = GKEStartJobOperator(
        task_id="job_fetch_llm_raw_data_bis",
        project_id=PROJECT_CONFIG.project_id,
        location=PROJECT_CONFIG.cluster_region,
        cluster_name=PROJECT_CONFIG.cluster_name,
        namespace="default",
        parallelism=1,
        image="google/cloud-sdk:latest",
        name="fetch-llm-raw-data-bis",
        cmds=[
            "curl",
            "-L",
            "-o",
            GCS_FT_DATASET_BIS_PATH,
            DATABRICKS_DOLLY_15K_JSONL,
        ],
        container_resources=build_resource_requirements("200m", "200Mi"),
        service_account_name="default",
        volume_mounts=GCS_VOLUME_MOUNT,
        volumes=GCS_VOLUME,
        annotations=GCS_ANNOTATION,
        wait_until_job_complete=True,
    )

    benchmark_env = {
        "CHAMPION_PATH": "/data/ft-champion",
        "STEPS": str(BENCHMARK_STEPS),
    } | {
        f"MODEL{task_id+1}_PATH": f"/data/gemma-finetuned-inference{task_id+1}"
        for task_id, _ in enumerate(FT_TASKS_ITEMS)
    }
    kueue_job_benchmark = GKEStartKueueJobOperator(
        task_id="kueue_job_benchmark",
        project_id=PROJECT_CONFIG.project_id,
        location=PROJECT_CONFIG.cluster_region,
        cluster_name=PROJECT_CONFIG.cluster_name,
        queue_name="local-queue",
        namespace="default",
        parallelism=1,
        image=BENCHMARK_APP_LATEST,
        name="llm-benchmark",
        suspend=True,
        container_resources=build_resource_requirements("6000m", "50Gi"),
        env_vars=benchmark_env,
        service_account_name="default",
        volume_mounts=GCS_VOLUME_MOUNT,
        volumes=GCS_VOLUME,
        annotations=GCS_ANNOTATION,
        wait_until_job_complete=True,
    )

    upload_model_task = UploadModelOperator(
        task_id="upload_model_to_vertex_ai",
        project_id=PROJECT_CONFIG.project_id,
        region=PROJECT_CONFIG.cluster_region,
        model={
            "display_name": "gemma-2b-en-ft-champion",
            "artifact_uri": f"gs://{GCS_BUCKET_NAME}/ft-champion",
            "container_spec": {
                "image_uri": VLLM_DOCKER_URI,
                "command": ["python", "-m", "vllm.entrypoints.api_server"],
                "args": VLLM_ARGS,
                "ports": [{"container_port": 7080}],
                "predict_route": "/generate",
                "health_route": "/ping",
            },
        },
    )

    create_endpoint_task = CreateEndpointOperator(
        task_id="create_endpoint",
        project_id=PROJECT_CONFIG.project_id,
        region=PROJECT_CONFIG.cluster_region,
        endpoint={
            "display_name": "gemma-2b-en-ft-champion-endpoint",
        },
    )

    deploy_model_task = DeployModelOperator(
        task_id="deploy_model",
        project_id=PROJECT_CONFIG.project_id,
        region=PROJECT_CONFIG.cluster_region,
        endpoint_id="{{ task_instance.xcom_pull(task_ids='create_endpoint')['name'].split('/')[-1] }}",
        traffic_split={"0": 100},
        deployed_model={
            "model": "{{ task_instance.xcom_pull(task_ids='upload_model_to_vertex_ai')['model'] }}",
            "display_name": "gemma-2b-en-ft-champion-vllm",
            "dedicated_resources": {
                "machine_spec": {
                    "machine_type": "g2-standard-12",
                    "accelerator_type": "NVIDIA_L4",
                    "accelerator_count": 1,
                },
                "min_replica_count": 1,
                "max_replica_count": 1,
            },
        },
    )

    # The below task doesn't really generate any reports
    # It's just a placeholder showcasing how these types of workflows usually end
    kueue_job_gen_report_bis = GKEStartKueueJobOperator(
        task_id="kueue_job_gen_report_bis",
        project_id=PROJECT_CONFIG.project_id,
        location=PROJECT_CONFIG.cluster_region,
        cluster_name=PROJECT_CONFIG.cluster_name,
        queue_name="local-queue",
        namespace="default",
        parallelism=1,
        image="polinux/stress",
        cmds=["stress", "-c", "1", "-t", "30"],
        name="generate-report",
        suspend=True,
        container_resources=build_resource_requirements("100m", "100Mi"),
    )

    delete_cluster = GKEDeleteClusterOperator(
        task_id="delete_cluster",
        name=PROJECT_CONFIG.cluster_name,
        project_id=PROJECT_CONFIG.project_id,
        location=PROJECT_CONFIG.cluster_region,
    )

    # setup cluster resources
    (
        create_cluster
        >> add_kueue_cluster
        >> create_tiny_resource_flavor
        >> create_highmem_resource_flavor
        >> create_cluster_queue
    )

    # setup team and job resources
    create_cluster_queue >> create_local_queue >> job_fetch_data

    # setup another team and job resources
    create_cluster_queue >> create_local_queue_bis >> job_fetch_data_bis

    # jobs team prime
    for job_id, job_steps in enumerate(FT_TASKS_ITEMS):
        (
            job_fetch_data
            >> build_kueue_preprocess_job(PROJECT_CONFIG, job_steps, job_id + 1)
            >> build_kueue_finetuning_job(PROJECT_CONFIG, job_steps, job_id + 1)
            >> build_kueue_convert_job(PROJECT_CONFIG, job_id + 1)
            >> kueue_job_benchmark
        )

    # jobs team bis
    for job_id, job_steps in enumerate(FT_TASKS_ITEMS):
        (
            job_fetch_data_bis
            >> build_kueue_preprocess_job(
                PROJECT_CONFIG,
                job_steps,
                job_id + 1,
                task_id="kueue_job_preprocess_bis",
                local_queue="local-queue-bis",
                input_dataset=GCS_FT_DATASET_BIS_PATH,
                output_file=fine_tuned_path_bis_template,
            )
            >> build_kueue_finetuning_job(
                PROJECT_CONFIG,
                job_steps,
                job_id + 1,
                task_id="kueue_job_finetune_bis",
                local_queue="local-queue-bis",
                input_ft_data_path=fine_tuned_path_bis_template,
                output_model_path=fine_tuned_model_path_bis_template,
            )
            >> kueue_job_gen_report_bis
        )

    # Deployment for team prime
    (
        kueue_job_benchmark
        >> upload_model_task
        >> create_endpoint_task
        >> deploy_model_task
    )

    # cleanup
    deploy_model_task >> delete_cluster
    kueue_job_gen_report_bis >> delete_cluster
