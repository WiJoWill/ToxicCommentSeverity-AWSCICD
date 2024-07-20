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
packages/button
├── lib
│   ├── button.d.ts
│   ├── button.js
│   ├── button.js.map
│   ├── button.stories.d.ts
│   ├── button.stories.js
│   ├── button.stories.js.map
│   ├── index.d.ts
│   ├── index.js
│   └── index.js.map
├── package.json
├── src
│   ├── button.stories.tsx
│   ├── button.tsx
│   └── index.ts
└── tsconfig.json


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

### `.DS_Store`
System file (usually hidden) created by macOS.

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


