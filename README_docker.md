# Docker Usage for Deformable DETR on USTC GPU Cluster

This project has been containerized for use on the USTC GPU Cluster.

## Prerequisites

- Access to the USTC GPU Cluster.
- Code copied to your home directory on the cluster (e.g., `/ghome/<username>/Deformable-DETR`).

## 1. Build the Docker Image

According to the cluster documentation, you should build the image on the `G101` node or locally and push it.

### Option A: Build on G101 (Recommended)

1.  Login to the cluster and request a debug node (G101) or use `startdocker` to get a shell where you can build (if allowed).
    *   *Note: The cluster docs mention building in `/ghome/<username>/dockertmp`.*
2.  Navigate to the project directory.
3.  Run the build script:
    ```bash
    bash build_docker.sh
    ```
    Or manually:
    ```bash
    docker build -t <your_username>/deformable_detr:v1 .
    ```

### Option B: Request Admin to Publish

If you cannot build it yourself, you can provide the `Dockerfile` to the admins as per the cluster documentation.

## 2. Submit a Job

Use a PBS script to submit a job using your built image.

Example `job.pbs`:

```bash
#PBS -N deformable_detr_train
#PBS -o /ghome/<username>/$PBS_JOBID.out
#PBS -e /ghome/<username>/$PBS_JOBID.err
#PBS -l nodes=1:gpus=1:S
#PBS -r y

cd $PBS_O_WORKDIR

# Replace <your_username>/deformable_detr:v1 with your actual image tag
# Replace <path_to_data> with your data directory (e.g. /gdata/<username>/coco)

startdocker -D /gdata/<username>/coco -c "python main.py --coco_path /gdata/<username>/coco" <your_username>/deformable_detr:v1
```

## Notes on Environment

- **Base Image**: `bit:5000/ubuntu18.04_cuda11.1_devel_cudnn8`
- **PyTorch**: 1.8.1 (Compatible with RTX 3090 and older cards)
- **CUDA**: 11.1
- **Custom Ops**: The C++ operators have been patched to compile with PyTorch 1.8+ and are compiled automatically during the Docker build.

## Troubleshooting

If you encounter "No kernel image is available for execution on the device", ensure you are using the image built with CUDA 11.1 (as defined in the Dockerfile) and not an older CUDA 10.1 image on RTX 3090 nodes.
