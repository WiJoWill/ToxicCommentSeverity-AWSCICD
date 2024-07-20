# Toxic Comments Severity

## Project Description

This is a deep learning project that leverages modern NLP techniques, specifically large language models (LLMs), to analyze the offensiveness and toxicity of comments in English. The project is inspired by the Kaggle Competition 'Jigsaw Rate Severity of Toxic Comments', and it includes the final model along with its deployment.

### Data Sources

The data used in this project comes from several sources, including:
- [Jigsaw Toxic Severity Rating Competition Data](https://www.kaggle.com/c/jigsaw-toxic-severity-rating/data)
- [Jigsaw Toxic Comment Classification Challenge (2018)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
- [Jigsaw Unintended Bias in Toxicity Classification (2019)](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)
- [Ruddit Dataset](https://www.kaggle.com/rajkumarl/ruddit-jigsaw-dataset)

All external data should be placed in the corresponding locations within the input folder.

### Model Features

The model predicts the severity of six features:
1. Toxic
2. Severe Toxic
3. Obscene
4. Threat
5. Insult
6. Identity Hate

The final score is calculated as a linear combination of these features, with a default weight distribution. This distribution can be modified in the `predict_pipeline` class.

### Model

The current project uses the DeBERTa model as the base model for predictions.


## Directory Structure

The directory structure of this project is as follows:
# Tree
```
ToxicCommentSeverity-AWSCICID/
├── .ebextensions/
├── .github/
├── artifacts/
│ ├── comments_to_score.csv
│ ├── config.json
│ ├── merges.txt
│ ├── model/
│ ├── special_tokens_map.json
│ ├── tokenizer_config.json
│ └── vocab.json
├── src/
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ ├── pipeline/
│ │ ├── init.py
│ │ ├── predict_pipeline.py
│ │ ├── train_pipeline.py
│ │ ├── exception.py
│ │ ├── logger.py
│ │ └── utils.py
├── templates/
│ ├── home.html
│ ├── home_style.css
│ └── index.html
├── .DS_Store
├── .gitattributes
├── .gitignore
├── Dockerfile
├── README.md
├── app.py
├── requirements.txt
└── setup.py
```

## Directory and File Descriptions

### `.ebextensions/`
Contains configuration files for AWS Elastic Beanstalk.

### `.github/`
Contains GitHub-specific files, such as workflows and actions for CI/CD.

### `artifacts/`
Contains various artifacts required for the project.
- `comments_to_score.csv`: Dataset containing comments to be scored.
- `config.json`: Configuration file for the project.
- `merges.txt`: Merges information for tokenization.
- `model/`: Directory containing the saved model files.
- `special_tokens_map.json`: Mapping of special tokens used in the model.
- `tokenizer_config.json`: Configuration for the tokenizer.
- `vocab.json`: Vocabulary file for the tokenizer.

### `src/`
Contains the main source code for the project.
- `components/`
  - `data_ingestion.py`: Script for data ingestion.
  - `data_transformation.py`: Script for data transformation.
  - `model_trainer.py`: Script for training the model.
- `pipeline/`
  - `__init__.py`: Initializes the pipeline module.
  - `predict_pipeline.py`: Script for making predictions.
  - `train_pipeline.py`: Script for training the pipeline.
  - `exception.py`: Custom exception handling.
  - `logger.py`: Logger configuration.
  - `utils.py`: Utility functions.

### `templates/`
Contains HTML templates for the web application.
- `home.html`: Home page template.
- `home_style.css`: CSS for styling the home page.
- `index.html`: Index page template.

### `.gitattributes`
Defines attributes for pathnames to customize the behavior of Git.

### `.gitignore`
Specifies files and directories that should be ignored by Git.

### `Dockerfile`
Defines the Docker container setup for the project.

### `README.md`
Provides an overview and documentation for the project.

### `app.py`
Main application script for running the project.

### `requirements.txt`
Lists the Python dependencies required for the project.

### `setup.py`
Script for setting up the Python package.

## Deployment with Docker and AWS EC2

To deploy this project using Docker and AWS EC2, follow these steps:

### Docker Setup on EC2

1. **Update and Upgrade Packages (Optional)**
    ```sh
    sudo apt-get update -y
    sudo apt-get upgrade
    ```

2. **Install Docker (Required)**
    ```sh
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    newgrp docker
    ```

3. **Configure the Runner Following GitHub**

   Follow the GitHub Actions runner setup guide to configure the runner on your EC2 instance.

### Adjust the Instance Setting
1. **Adjust the Instance Security Groups**
    ```sh
    Remember to set the appropriate port in Security Groups
    ```

This will deploy the application on your EC2 instance, accessible through the public IP address of the instance.

## Contributing

Thanks for the youtuber: https://www.youtube.com/@krishnaik06
Thanks for the competition: https://www.kaggle.com/c/jigsaw-toxic-severity-rating

## License

MIT License
