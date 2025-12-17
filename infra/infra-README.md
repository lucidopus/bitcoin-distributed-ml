# Infrastructure Scripts

This folder contains the shell scripts used to manage our Google Cloud Dataproc infrastructure. This is the control center for our distributed computing environment and interacting with the cluster.

## File Descriptions

- **`create_cluster.sh`**: This script initializes our Dataproc cluster. It sets up the master and worker nodes, configures the machine types, and prepares the environment for Spark jobs.
- **`delete_cluster.sh`**: A utility script to cleanly tear down the cluster (to ensure we aren't incurring unnecessary costs when the experiments are done).
- **`submit-job.sh`**: This is our primary interface for interacting with the cluster. It abstracts away the complex `gcloud dataproc jobs submit` commands, allowing us to easily submit Python scripts with various arguments and common configurations.
