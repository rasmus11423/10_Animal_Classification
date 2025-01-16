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
