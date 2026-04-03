# MLFLOW--EC2

## Project overview
- Integrate ml model training with MLFLLOW in AWS EC2 + dockerized container.
- The EC2 instance used is PH-Vector-Database.

## Project description
- Theere are 5 config yml files which can be changed.
- they are
    - config/data/data_v1.yml
    - config/database/db_v1.yml
    - config/features/features_v1.yml
    - config/ml/ml_v1.yml
    - config/search_spaces/search_spaces_v1.yml
- Rest of env.py, filepaths.py remain constant.
- I have already set up the mlflow server on port 5000
  
## How to run
- Login to the EC2 instance using ssh.
- I have already kept the mlflow-server running.
- Run the pipeline thorugh docker compose
```
docker-compose up ml-pipeline
```

## Screenshots
- **Experiments:** http://{ec2-private-ip}:5000/#/experiments
<img width="1452" height="393" alt="image" src="https://github.com/user-attachments/assets/1c593a8c-5820-4203-bd19-b145dabcfdf7" />

- **Runs:**
<img width="1797" height="617" alt="image" src="https://github.com/user-attachments/assets/aa66dc1e-85a5-4e8c-9e06-2c07476ca57b" />

- **Artifacts:** http://{ec2-private-ip}:5000/#/experiments/{experiment_id}/runs/{run_id}/artifacts
 <img width="1422" height="624" alt="image" src="https://github.com/user-attachments/assets/8bc58556-70b1-405f-9861-412a171a116a" />

## Future improvements
- Migrate artifact store to S3 (as artifacts cannot be accessed outside of EC2 instance)
- Replace SQLite with PostgreSQL 
