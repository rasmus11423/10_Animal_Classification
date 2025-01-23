# How to run

The project requires (kaggle authentication)[https://www.kaggle.com/docs/api] to download the dataset. ("Go to the 'Account' tab of your user profile and select 'Create New Token'. This will trigger the download of kaggle.json, a file containing your API credentials.")

## General comments on how to run
- Config files are present in 'configs/'         
    - 'training_configs.yaml': Define the default configurations of a single training run - including: hyperparameters(batch_size, epochs), optimizer(name, lr), criterion). The parameters in the configuration can be overwritten by parsing arguments into the command line.
    - 'evaluate_configs.yaml': Defines the default parameters of the evaluation of a model - including: hyperparameters(batch_size). The parameters can be overwritten by parsing arguments into the command line.
    - 'sweep.yaml': Defines a hyperparameter sweep experiment with specific ranges distributions, or lists of hyperparameters to explore (e.g., lr: log_uniform or epochs: [10, 15, 25]). It also includes the method for exploring the hyperparameter space, such as random, grid, or bayesian, as well as a metric to optimize. *To run sweep.yaml in your machine you must change the command to run your own env* (MAYBE MAKE THIS DYNAMIC WITH DOCKER FILE?)
    - 'gcloudconfig.yaml': Defines the configuration for the Google Cloud Vertex AI training job.


to run the training job on vertex ai, you must call this command:
'''
gcloud builds submit --config=configs/vertex_ai_train.yaml .
'''
Remember to push the changes to main, so that the docker image is updated. Otherwise you will need to manually update the image in the vertex ai training job by running: 

'''
gcloud builds submit . --config=cloudbuild_train.yaml
'''


to deploy the api, you must call this command:

'''
gcloud builds submit . --config=cloudbuild_api.yaml
'''



## Features
- (SOMETHING ABOUT W&B)
- The project features a torch profiler, where each training is logged within 'runs/profiler_logs' and can be accessed via 'tensorboard --logdir=runs/profiler_logs'. Profilers are used to identify bottlenecks in the training process and optimize the model's performance. For instance, it identifies the most time-consuming operations in the training process, such as data loading, forward and backward passes, and optimizer steps.
- The project features a wandb logger, where each training is logged within 'runs/wandb' and can be accessed via 'wandb.init()'. The logger is used to log the training process, including the loss, accuracy, and images along with the model's predictions and true labels.
- github actions: 
    * unit tests
    * linting
    * type checking
    * dependabot


# 10_Animal_Classification
This is the project work for group 44 in the course: Machine Learning Operations at DTU. This group consists of: Rasmus Laansalu, Marcos Bauch Mira, Viraj Rajurkar, Anke van de Watering, Abrahim Abbas. 

1. **Overall Goal:** The goal is to classify images of animals into ten categories (dog, horse, cat, spider, butterfly, chicken, sheep, cow, squirrel, and elephant). The project involves building and evaluating deep learning models to achieve high classification accuracy while exploring reproducibility and scalability.
   
2. **Framework:**  As a starting point, we intend to use the composer framework in order to speed up the training time and enhance the entire workflow. This will be enable us to run several expierments and allow for fast and efficient hyperparameter grid-search. In the mean time we will also investigate the possibility of using frameworks such as ONNX to optimize and speed up the inference time. 
4. **Data:** The dataset consists of 28K medium-quality animal images, with labels for each of the ten categories of animals the picture belongs to (dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, elephant). The images were originally taken from Google Images, and the animal labels have been checked by humans. There are not equal numbers of each image, but they all range from 2,000 to 5,000 units.
6. **Deep Learning models:** To classify the images we are implementing a Convolutional Neural Network while incorporating PyTorch image models (TIMM). 
# animal_classification

banas

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
