# 'Push' Model Prediction API

This repository contains API for training, scoring, and making predictions with a machine learning model made for predicting the carrier IDs based on the manifest and stops data. The API includes endpoints for managing the model lifecycle and supports flexible input formats.


## Features

- **Model Training**: Train a machine learning model with uploaded data.
- **Model Scoring**: Evaluate the model's performance on test data.
- **Prediction**: Generate predictions for input data, supporting both:
  - List of objects
  - Dictionary of lists (column-oriented format)

## Requirements

### General:
- Python 3.10+
- nginx
- tmux

### Python-specific:
- Flask
- numpy
- pyarrow
- pandas
- scikit-learn
- Gunicorn

*These are present in `requirements.txt`.*

## Installation

### 1. Install Python 3.10 with `pip` and `venv`

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

### 2. Install `nginx` and `tmux`

```bash
sudo apt install nginx
sudo apt install tmux
```

### 3. Clone the repository and navigate to the project directory

```bash
git clone <repository_url>
cd <repository_name>
```

### 4. Create and activate a virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```


## Nginx Configuration

### 1. Default `nginx` Configuration Backup (Optional)

Before configuring a new server, you can back up the default `nginx` configuration:

```bash
sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup
```

### 2. Create a New Nginx Configuration

Create a new configuration file for the API:

```bash
sudo nano /etc/nginx/sites-available/myapp
```

Add the following content:

```nginx
server {
    listen 80;
    server_name _; # or replace _ with the public IP

    # Proxy all other requests to Gunicorn
    location / {
        proxy_pass http://0.0.0.0:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Logs
    error_log /var/log/nginx/myapp_error.log;
    access_log /var/log/nginx/myapp_access.log;
}
```

### 3. Link the Configuration and Restart Nginx

Link the new configuration to the `sites-enabled` directory and restart `nginx`:

```bash
sudo ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

## Running the Application

### 1. Activate the Virtual Environment

*Inside the push module folder with virtual env.*

```bash
source venv/bin/activate
```

### 2. Use `tmux` to Keep the Application Running

Start a new `tmux` session:

```bash
tmux new -s myapp
```

To enter an existing session:

```bash
tmux attach -t myapp
```

Inside the `tmux` session, run the application using Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:80 wsgi:app
```

To detach from the `tmux` session without stopping the application, press:

```bash
Ctrl + B, then D
```

To reattach to the session later:

```bash
tmux attach -t myapp
```

### 3. Verify the Application

Access the API in your browser or via a tool like Postman at:

```bash
http://<your_public_ip>/
```


## Endpoints Overview

Please find the detailed endpoints overview in `Swagger.yaml` provided in the repository.

## Folder Structure

```
.
├── app.py                   # Main API script
├── requirements.txt         # Required Python packages
├── checkpoints/             # Saved models and data controllers
│   ├── data_controller.pkl  # (Generated as the model is trained)
│   ├── model_controller.pkl # (Generated as the model is trained)
├── dataset/                 # Dataset files should be transfered here (e.g., example_data.parquet)
├── swagger.yaml             # OpenAPI 3.0 specification file
├── wsgi.py                  # Gunicorn entry point
└── README.md                # Project documentation
```

---

### Additional Notes

- Ensure the IP address in the `nginx` configuration matches your server's public IP.
- If running on a cloud service, ensure the firewall allows traffic on port `80`.