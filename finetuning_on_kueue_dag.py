from airflow import models
from kubernetes.client import models as k8s
from airflow.providers.google.cloud.operators.kubernetes_engine import (
    GKECreateClusterOperator,
    GKEDeleteClusterOperator,
    GKEStartKueueInsideClusterOperator,
    GKECreateCustomResourceOperator,
    GKEStartKueueJobOperator,
    GKEStartJobOperator,
)
from airflow.utils.dates import days_ago

from kubernetes.client import V1ResourceRequirements, V1Volume, V1VolumeMount, V1CSIVolumeSource


with models.DAG(
    "kueue_highmem_machines_preprocessing_finetuning_testing",
    schedule_interval=None,  
    start_date=days_ago(1),
    tags=["example"],
) as dag:
    PROJECT_ID = "[PLACEHOLDER]"
    WORKLOAD_IDENTITY_POOL = PROJECT_ID + ".svc.id.goog"
    CLUSTER_REGION = "us-central1"
    CLUSTER_NAME = "gke-kueue-cluster"
    CLUSTER = {
        "name": CLUSTER_NAME,
        "workload_identity_config": {
                "workload_pool": WORKLOAD_IDENTITY_POOL
            },
        "addons_config": {
                "gcs_fuse_csi_driver_config": {
                    'enabled': True
                }
            },
        "node_pools": [
            {
                'name': 'autoscaled-pool-tiny',
                'initialNodeCount': 1,
                'autoscaling': {
                    'enabled': True,
                    'minNodeCount': 1,  
                    'maxNodeCount': 3,  
                'config': {
                    'labels': {
                        'workflow-type': 'tiny',
                    },
                    'workload_metadata_config': {
                        'mode': 'GKE_METADATA'
                    }
                },
            },
            {
                'name': 'autoscaled-pool-highmem',
                'initialNodeCount': 1,
                'autoscaling': {
                    'enabled': True,
                    'minNodeCount': 1,
                    'maxNodeCount': 1, 
                },
                'config': {
                    'machine_type': 'n4-standard-16',  
                    'labels': {
                        'workflow-type': 'high-mem',
                    },
                    'workload_metadata_config': {
                        'mode': 'GKE_METADATA'
                    }
                },
                'locations': ['us-central1-a'] 
            },
            {
                'name': 'autoscaled-pool-mediummem',
                'initialNodeCount': 0,
                'autoscaling': {
                    'enabled': True,
                    'minNodeCount': 0,
                    'maxNodeCount': 2, 
                },
                'config': {
                    'machine_type': 'n4-standard-8',  # Specify the machine type
                    'labels': {
                        'workflow-type': 'high-mem',
                    },
                    'workload_metadata_config': {
                        'mode': 'GKE_METADATA'
                    }
                },
                'locations': ['us-central1-a']  # Specify the zone for this node pool
            },
        ]
    }
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

    create_cluster = GKECreateClusterOperator(
        task_id="create_cluster",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        body=CLUSTER,
    )

    add_kueue_cluster = GKEStartKueueInsideClusterOperator(
        task_id="add_kueue_cluster",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        kueue_version="v0.8.0",
    )

    create_tiny_resource_flavor = GKECreateCustomResourceOperator(
        task_id="create_tiny_resource_flavor",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        yaml_conf=tiny_flavor_conf,
        custom_resource_definition=True,
        namespaced=False,
    )

    create_highmem_resource_flavor = GKECreateCustomResourceOperator(
        task_id="create_highmem_resource_flavor",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        yaml_conf=highmem_flavor_conf,
        custom_resource_definition=True,
        namespaced=False,
    )

    create_cluster_queue = GKECreateCustomResourceOperator(
        task_id="create_cluster_queue",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        yaml_conf=cluster_conf,
        custom_resource_definition=True,
        namespaced=False,
    )

    create_local_queue = GKECreateCustomResourceOperator(
        task_id="create_local_queue",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        yaml_conf=local_conf,
        custom_resource_definition=True,
    )

    job_fetch_data = GKEStartJobOperator(
        task_id="job_fetch_llm_raw_data",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        namespace="default",
        parallelism=1,
        image="google/cloud-sdk:latest",
        name="fetch-llm-raw-data",
        cmds=[
            "curl",
            "-L",
            "-o", 
            "/data/databricks-dolly-15k.jsonl", 
            "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
        ],
        container_resources=V1ResourceRequirements(
            requests={
                "cpu": "200m",
                "memory": "200Mi",
            },
            limits={
                "cpu": "200m",
                "memory": "200Mi",
            },
        ),
        
        service_account_name="default",
        volume_mounts=[
            V1VolumeMount(
                name="gcs-fuse-csi",
                mount_path="/data",
            ),
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse-csi",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": "example-model-trainings",
                    },
                ),
            ),
        ],
        annotations={
            "gke-gcsfuse/volumes": "true",
        },
        wait_until_job_complete = True,
    )

    # [START howto_operator_kueue_start_job]
    kueue_job_preprocess1 = GKEStartKueueJobOperator(
        task_id="kueue_job_preprocess1",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        queue_name="local-queue",
        namespace="default",
        parallelism=1,
        image=".../llm-preprocessing-app:latest",
        name="llm-data-preprocessing",
        suspend=True,
        container_resources=V1ResourceRequirements(
            requests={
                "cpu": "200m",
                "memory": "200Mi",
            },
            limits={
                "cpu": "200m",
                "memory": "200Mi",
            },
        ),
        env_vars={
            "INPUT_FILE": "/data/databricks-dolly-15k.jsonl",
            "OUTPUT_FILE": "/data/fine-tuned1.jsonl",
            "NUM_ITEMS": "1000",
        },
        service_account_name="default",
        volume_mounts=[
            V1VolumeMount(
                name="gcs-fuse-csi",
                mount_path="/data",
            ),
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse-csi",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": "example-model-trainings",
                    },
                ),
            ),
        ],
        annotations={
            "gke-gcsfuse/volumes": "true",
        },
        wait_until_job_complete = True,
    )

    kueue_job_preprocess2 = GKEStartKueueJobOperator(
        task_id="kueue_job_preprocess2",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        queue_name="local-queue",
        namespace="default",
        parallelism=1,
        image=".../polish-punk-depot/llm-preprocessing-app:latest",
        name="llm-data-preprocessing",
        suspend=True,
        container_resources=V1ResourceRequirements(
            requests={
                "cpu": "200m",
                "memory": "200Mi",
            },
            limits={
                "cpu": "200m",
                "memory": "200Mi",
            },
        ),
        env_vars={
            "INPUT_FILE": "/data/databricks-dolly-15k.jsonl",
            "OUTPUT_FILE": "/data/fine-tuned2.jsonl",
            "NUM_ITEMS": "1000",
        },
        service_account_name="default",
        volume_mounts=[
            V1VolumeMount(
                name="gcs-fuse-csi",
                mount_path="/data",
            ),
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse-csi",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": "example-model-trainings",
                    },
                ),
            ),
        ],
        annotations={
            "gke-gcsfuse/volumes": "true",
        },
        wait_until_job_complete = True,
    )

    kueue_job_preprocess3 = GKEStartKueueJobOperator(
        task_id="kueue_job_preprocess3",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        queue_name="local-queue",
        namespace="default",
        parallelism=1,
        image=".../llm-preprocessing-app:latest",
        name="llm-data-preprocessing",
        suspend=True,
        container_resources=V1ResourceRequirements(
            requests={
                "cpu": "200m",
                "memory": "200Mi",
            },
            limits={
                "cpu": "200m",
                "memory": "200Mi",
            },
        ),
        env_vars={
            "INPUT_FILE": "/data/databricks-dolly-15k.jsonl",
            "OUTPUT_FILE": "/data/fine-tuned3.jsonl",
            "NUM_ITEMS": "1000",
        },
        service_account_name="default",
        volume_mounts=[
            V1VolumeMount(
                name="gcs-fuse-csi",
                mount_path="/data",
            ),
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse-csi",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": "example-model-trainings",
                    },
                ),
            ),
        ],
        annotations={
            "gke-gcsfuse/volumes": "true",
        },
        wait_until_job_complete = True,
    )

    kueue_job_preprocess4 = GKEStartKueueJobOperator(
        task_id="kueue_job_preprocess4",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        queue_name="local-queue",
        namespace="default",
        parallelism=1,
        image=".../llm-preprocessing-app:latest",
        name="llm-data-preprocessing",
        suspend=True,
        container_resources=V1ResourceRequirements(
            requests={
                "cpu": "200m",
                "memory": "200Mi",
            },
            limits={
                "cpu": "200m",
                "memory": "200Mi",
            },
        ),
        env_vars={
            "INPUT_FILE": "/data/databricks-dolly-15k.jsonl",
            "OUTPUT_FILE": "/data/fine-tuned4.jsonl",
            "NUM_ITEMS": "1000",
        },
        service_account_name="default",
        volume_mounts=[
            V1VolumeMount(
                name="gcs-fuse-csi",
                mount_path="/data",
            ),
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse-csi",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": "example-model-trainings",
                    },
                ),
            ),
        ],
        annotations={
            "gke-gcsfuse/volumes": "true",
        },
        wait_until_job_complete = True,
    )

    kueue_job_fine_tune1 = GKEStartKueueJobOperator(
        task_id="kueue_job_finetune1",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        queue_name="local-queue",
        namespace="default",
        parallelism=1,
        image=".../llm-finetuning-app:latest",
        name="llm-data-finetuning",
        suspend=True,
        container_resources=V1ResourceRequirements(
            requests={
                "cpu": "7000m",
                "memory": "20Gi",
            },
            limits={
                "cpu": "7000m",
                "memory": "25Gi",
            },
        ),
        env_vars={
            "FILE_PATH": "/data/fine-tuned1.jsonl",
            "OUTPUT_FILE_NAME": "/data/gemma-finetuned1",
            "STEPS": "1",
            "KAGGLE_USERNAME": "[PLACEHOLDER_KAGGLE_USERNAME]",
            "KAGGLE_KEY": "[PLACEHOLDER_KAGGLE_PASSWORD]"
        },
        service_account_name="default",
        volume_mounts=[
            V1VolumeMount(
                name="gcs-fuse-csi",
                mount_path="/data",
            ),
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse-csi",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": "example-model-trainings",
                    },
                ),
            ),
        ],
        annotations={
            "gke-gcsfuse/volumes": "true",
        },
        wait_until_job_complete = True,
    )

    kueue_job_fine_tune2 = GKEStartKueueJobOperator(
        task_id="kueue_job_finetune2",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        queue_name="local-queue",
        namespace="default",
        parallelism=1,
        image=".../llm-finetuning-app:latest",
        name="llm-data-finetuning",
        suspend=True,
        container_resources=V1ResourceRequirements(
            requests={
                "cpu": "7000m",
                "memory": "20Gi",
            },
            limits={
                "cpu": "7000m",
                "memory": "25Gi",
            },
        ),
        env_vars={
            "FILE_PATH": "/data/fine-tuned2.jsonl",
            "OUTPUT_FILE_NAME": "/data/gemma-finetuned2",
            "STEPS": "2",
            "KAGGLE_USERNAME": "[PLACEHOLDER_KAGGLE_USERNAME]",
            "KAGGLE_KEY": "[PLACEHOLDER_KAGGLE_PASSWORD]"
        },
        service_account_name="default",
        volume_mounts=[
            V1VolumeMount(
                name="gcs-fuse-csi",
                mount_path="/data",
            ),
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse-csi",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": "example-model-trainings",
                    },
                ),
            ),
        ],
        annotations={
            "gke-gcsfuse/volumes": "true",
        },
        wait_until_job_complete = True,
    )

    kueue_job_fine_tune3 = GKEStartKueueJobOperator(
        task_id="kueue_job_finetune3",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        queue_name="local-queue",
        namespace="default",
        parallelism=1,
        image=".../llm-finetuning-app:latest",
        name="llm-data-finetuning",
        suspend=True,
        container_resources=V1ResourceRequirements(
            requests={
                "cpu": "7000m",
                "memory": "20Gi",
            },
            limits={
                "cpu": "7000m",
                "memory": "25Gi",
            },
        ),
        env_vars={
            "FILE_PATH": "/data/fine-tuned3.jsonl",
            "OUTPUT_FILE_NAME": "/data/gemma-finetuned3",
            "STEPS": "3",
            "KAGGLE_USERNAME": "[PLACEHOLDER_KAGGLE_USERNAME]",
            "KAGGLE_KEY": "[PLACEHOLDER_KAGGLE_PASSWORD]"
        },
        service_account_name="default",
        volume_mounts=[
            V1VolumeMount(
                name="gcs-fuse-csi",
                mount_path="/data",
            ),
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse-csi",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": "example-model-trainings",
                    },
                ),
            ),
        ],
        annotations={
            "gke-gcsfuse/volumes": "true",
        },
        wait_until_job_complete = True,
    )

    kueue_job_fine_tune4 = GKEStartKueueJobOperator(
        task_id="kueue_job_finetune4",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        queue_name="local-queue",
        namespace="default",
        parallelism=1,
        image=".../llm-finetuning-app:latest",
        name="llm-data-finetuning",
        suspend=True,
        container_resources=V1ResourceRequirements(
            requests={
                "cpu": "7000m",
                "memory": "20Gi",
            },
            limits={
                "cpu": "7000m",
                "memory": "25Gi",
            },
        ),
        env_vars={
            "FILE_PATH": "/data/fine-tuned4.jsonl",
            "OUTPUT_FILE_NAME": "/data/gemma-finetuned4",
            "STEPS": "4",
            "KAGGLE_USERNAME": "[PLACEHOLDER_KAGGLE_USERNAME]",
            "KAGGLE_KEY": "[PLACEHOLDER_KAGGLE_PASSWORD]"
        },
        service_account_name="default",
        volume_mounts=[
            V1VolumeMount(
                name="gcs-fuse-csi",
                mount_path="/data",
            ),
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse-csi",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": "example-model-trainings",
                    },
                ),
            ),
        ],
        annotations={
            "gke-gcsfuse/volumes": "true",
        },
        wait_until_job_complete = True,
    )

    kueue_job_gen_report = GKEStartKueueJobOperator(
        task_id="kueue_job_gen_report",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        queue_name="local-queue",
        namespace="default",
        parallelism=1,
        image="polinux/stress",
        cmds=["stress", "-c", "1", "-t", "30"],
        name="generate-report",
        suspend=True,
        container_resources=k8s.V1ResourceRequirements(
            requests={
                "cpu": "100m",
                "memory": "100Mi",
            },
            limits={
                "cpu": "100m",
                "memory": "100Mi",
            },
        ),
    )

    jobs = [
        {
            "fetching": job_fetch_data,
            "preprocessing": kueue_job_preprocess1,
            "finetuning": kueue_job_fine_tune1,
            "reporting": kueue_job_gen_report,
        },
        {
            "fetching": job_fetch_data,
            "preprocessing": kueue_job_preprocess2,
            "finetuning": kueue_job_fine_tune2,
            "reporting": kueue_job_gen_report,
        },
        {
            "fetching": job_fetch_data,
            "preprocessing": kueue_job_preprocess3,
            "finetuning": kueue_job_fine_tune3,
            "reporting": kueue_job_gen_report,
        },
        {
            "fetching": job_fetch_data,
            "preprocessing": kueue_job_preprocess4,
            "finetuning": kueue_job_fine_tune4,
            "reporting": kueue_job_gen_report,
        },
    ]

    delete_cluster = GKEDeleteClusterOperator(
        task_id="delete_cluster",
        name=CLUSTER_NAME,
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
    )


    create_local_queue_bis = GKECreateCustomResourceOperator(
        task_id="create_local_queue_bis",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        yaml_conf=local_conf_bis,
        custom_resource_definition=True,
    )

    job_fetch_data_bis = GKEStartJobOperator(
        task_id="job_fetch_llm_raw_data_bis",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        namespace="default",
        parallelism=1,
        image="google/cloud-sdk:latest",
        name="fetch-llm-raw-data-bis",
        cmds=[
            "curl",
            "-L",
            "-o", 
            "/data/databricks-dolly-15k.jsonl", 
            "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
        ],
        container_resources=V1ResourceRequirements(
            requests={
                "cpu": "200m",
                "memory": "200Mi",
            },
            limits={
                "cpu": "200m",
                "memory": "200Mi",
            },
        ),
        
        service_account_name="default",
        volume_mounts=[
            V1VolumeMount(
                name="gcs-fuse-csi",
                mount_path="/data",
            ),
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse-csi",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": "example-model-trainings",
                    },
                ),
            ),
        ],
        annotations={
            "gke-gcsfuse/volumes": "true",
        },
        wait_until_job_complete = True,
    )

    # [START howto_operator_kueue_start_job]
    kueue_job_preprocess_bis1 = GKEStartKueueJobOperator(
        task_id="kueue_job_preprocess_bis1",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        queue_name="local-queue-bis",
        namespace="default",
        parallelism=1,
        image=".../llm-preprocessing-app:latest",
        name="llm-data-preprocessing-bis",
        suspend=True,
        container_resources=V1ResourceRequirements(
            requests={
                "cpu": "200m",
                "memory": "200Mi",
            },
            limits={
                "cpu": "200m",
                "memory": "200Mi",
            },
        ),
        env_vars={
            "INPUT_FILE": "/data/databricks-dolly-15k.jsonl",
            "OUTPUT_FILE": "/data/fine-tuned1.jsonl",
            "NUM_ITEMS": "1000",
        },
        service_account_name="default",
        volume_mounts=[
            V1VolumeMount(
                name="gcs-fuse-csi",
                mount_path="/data",
            ),
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse-csi",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": "example-model-trainings",
                    },
                ),
            ),
        ],
        annotations={
            "gke-gcsfuse/volumes": "true",
        },
        wait_until_job_complete = True,
    )

    kueue_job_preprocess_bis2 = GKEStartKueueJobOperator(
        task_id="kueue_job_preprocess_bis2",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        queue_name="local-queue-bis",
        namespace="default",
        parallelism=1,
        image=".../llm-preprocessing-app:latest",
        name="llm-data-preprocessing-bis",
        suspend=True,
        container_resources=V1ResourceRequirements(
            requests={
                "cpu": "200m",
                "memory": "200Mi",
            },
            limits={
                "cpu": "200m",
                "memory": "200Mi",
            },
        ),
        env_vars={
            "INPUT_FILE": "/data/databricks-dolly-15k.jsonl",
            "OUTPUT_FILE": "/data/fine-tuned2.jsonl",
            "NUM_ITEMS": "1000",
        },
        service_account_name="default",
        volume_mounts=[
            V1VolumeMount(
                name="gcs-fuse-csi",
                mount_path="/data",
            ),
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse-csi",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": "example-model-trainings",
                    },
                ),
            ),
        ],
        annotations={
            "gke-gcsfuse/volumes": "true",
        },
        wait_until_job_complete = True,
    )


    kueue_job_fine_tune_bis1 = GKEStartKueueJobOperator(
        task_id="kueue_job_finetune_bis1",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        queue_name="local-queue-bis",
        namespace="default",
        parallelism=1,
        image=".../llm-finetuning-app:latest",
        name="llm-data-finetuning-bis",
        suspend=True,
        container_resources=V1ResourceRequirements(
            requests={
                "cpu": "7000m",
                "memory": "20Gi",
            },
            limits={
                "cpu": "7000m",
                "memory": "25Gi",
            },
        ),
        env_vars={
            "FILE_PATH": "/data/fine-tuned1.jsonl",
            "OUTPUT_FILE_NAME": "/data/gemma-finetuned1",
            "STEPS": "1",
            "KAGGLE_USERNAME": "[PLACEHOLDER_KAGGLE_USERNAME]",
            "KAGGLE_KEY": "[PLACEHOLDER_KAGGLE_PASSWORD]"
        },
        service_account_name="default",
        volume_mounts=[
            V1VolumeMount(
                name="gcs-fuse-csi",
                mount_path="/data",
            ),
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse-csi",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": "example-model-trainings",
                    },
                ),
            ),
        ],
        annotations={
            "gke-gcsfuse/volumes": "true",
        },
        wait_until_job_complete = True,
    )

    kueue_job_fine_tune_bis2 = GKEStartKueueJobOperator(
        task_id="kueue_job_finetune_bis2",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        queue_name="local-queue-bis",
        namespace="default",
        parallelism=1,
        image=".../llm-finetuning-app:latest",
        name="llm-data-finetuning-bis",
        suspend=True,
        container_resources=V1ResourceRequirements(
            requests={
                "cpu": "7000m",
                "memory": "20Gi",
            },
            limits={
                "cpu": "7000m",
                "memory": "25Gi",
            },
        ),
        env_vars={
            "FILE_PATH": "/data/fine-tuned2.jsonl",
            "OUTPUT_FILE_NAME": "/data/gemma-finetuned2",
            "STEPS": "2",
            "KAGGLE_USERNAME": "[PLACEHOLDER_KAGGLE_USERNAME]",
            "KAGGLE_KEY": "[PLACEHOLDER_KAGGLE_PASSWORD]"
        },
        service_account_name="default",
        volume_mounts=[
            V1VolumeMount(
                name="gcs-fuse-csi",
                mount_path="/data",
            ),
        ],
        volumes=[
            V1Volume(
                name="gcs-fuse-csi",
                csi=V1CSIVolumeSource(
                    driver="gcsfuse.csi.storage.gke.io",
                    volume_attributes={
                        "bucketName": "example-model-trainings",
                    },
                ),
            ),
        ],
        annotations={
            "gke-gcsfuse/volumes": "true",
        },
        wait_until_job_complete = True,
    )

    kueue_job_gen_report_bis = GKEStartKueueJobOperator(
        task_id="kueue_job_gen_report_bis",
        project_id=PROJECT_ID,
        location=CLUSTER_REGION,
        cluster_name=CLUSTER_NAME,
        queue_name="local-queue",
        namespace="default",
        parallelism=1,
        image="polinux/stress",
        cmds=["stress", "-c", "1", "-t", "30"],
        name="generate-report",
        suspend=True,
        container_resources=k8s.V1ResourceRequirements(
            requests={
                "cpu": "100m",
                "memory": "100Mi",
            },
            limits={
                "cpu": "100m",
                "memory": "100Mi",
            },
        ),
    )

    jobs_bis = [
        {
            "fetching": job_fetch_data_bis,
            "preprocessing": kueue_job_preprocess_bis1,
            "finetuning": kueue_job_fine_tune_bis1,
            "reporting": kueue_job_gen_report_bis,
        },
        {
            "fetching": job_fetch_data_bis,
            "preprocessing": kueue_job_preprocess_bis2,
            "finetuning": kueue_job_fine_tune_bis2,
            "reporting": kueue_job_gen_report_bis,
        }
    ]



    # setup cluster resources
    create_cluster >> add_kueue_cluster  >> create_tiny_resource_flavor >> create_highmem_resource_flavor >> create_cluster_queue 


    # setup team and job resources
    create_cluster_queue >> create_local_queue >> job_fetch_data

    # jobs
    for job in jobs:
        job["fetching"] >> job["preprocessing"] >> job["finetuning"] >> job["reporting"]


    # setup team and job resources
    create_cluster_queue >> create_local_queue_bis >> job_fetch_data_bis

    # jobs
    for job in jobs_bis:
        job["fetching"] >> job["preprocessing"] >> job["finetuning"] >> job["reporting"]

    # cleanup
    kueue_job_gen_report >> delete_cluster
    kueue_job_gen_report_bis >> delete_cluster