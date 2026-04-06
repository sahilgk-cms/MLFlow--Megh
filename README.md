# MLFLOW--EC2

## Project overview
- This project integrates a machine learning training pipeline with **MLflow** for experiment tracking, model registry, and artifact management.
- The system is deployed on an **AWS EC2 instance** and containerized using **Docker Compose** to ensure reproducibility and ease of execution.

## Architecture Summary

- **MLflow Tracking Server** → Running on EC2 (Port 5000)
- **Pipeline Execution** → Dockerized (`ml-pipeline` service)
- **Artifact Store** → Local filesystem (`file:///mlflow/artifacts`)
- **Backend Store** → SQLite (current setup)

## Configuration
The pipeline is driven by configurable YAML files:
- `config/data/data_v1.yml`
- `config/database/db_v1.yml`
- `config/features/features_v1.yml`
- `config/ml/ml_v1.yml`
- `config/search_spaces/search_spaces_v1.yml`
### Notes:
- These files control data processing, feature engineering, and model training.
- Core utility files (`env.py`, `filepaths.py`) remain constant.
  
## How to run
- Login to the EC2 instance using ssh.
```
ssh <user>@<ec2-private-ip>
```
- Ensure MLflow Server is Running on http://<ec2-private-ip>:5000
- 
- Run the pipeline thorugh docker compose
```
docker-compose up app
```

## Screenshots
- **Experiments:** http://{ec2-private-ip}:5000/#/experiments
<img width="1452" height="393" alt="image" src="https://github.com/user-attachments/assets/1c593a8c-5820-4203-bd19-b145dabcfdf7" />

- **Runs:**
<img width="1797" height="617" alt="image" src="https://github.com/user-attachments/assets/aa66dc1e-85a5-4e8c-9e06-2c07476ca57b" />

- **Artifacts:** http://{ec2-private-ip}:5000/#/experiments/{experiment_id}/runs/{run_id}/artifacts
 <img width="1422" height="624" alt="image" src="https://github.com/user-attachments/assets/8bc58556-70b1-405f-9861-412a171a116a" />

- **Model Artifacts:** http://{ec2-private-ip}:5000/#/experiments/{experiment_id}/models/{model_id}/artifacts
<img width="1357" height="785" alt="image" src="https://github.com/user-attachments/assets/d01e15a7-e204-4564-9ad2-da9671ff6b08" />

- **Model Registry:**  http://{ec2-private-ip}:5000/#/models/{registered_model_name}
<img width="1370" height="825" alt="image" src="https://github.com/user-attachments/assets/776667c4-6d68-4d20-be51-f3511c348ae0" />

## Current limitations
- Artifacts are only accessible within the EC2 instance
- Cannot load models or download artifacts from local machines
- Limits integration with external systems

## Future improvements
- Migrate artifact store to S3 (as artifacts cannot be accessed outside of EC2 instance)
- Replace SQLite with PostgreSQL 
