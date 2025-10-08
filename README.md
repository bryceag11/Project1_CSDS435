# CSDS Project 1 

## File Structure 
```
Project
|   .gitignore
|   main.py
|   README.md
|
+---data
|       project1_dataset1.txt
|       project1_dataset2.txt # Categorical dataset
|
+---results # results and plots for all models
|   |   dataset1_results.csv 
|   |   dataset2_results.csv
|   |
|   \---plots
\---src
    |   classifiers.py # Non-neural network classifier initialization
    |   config.py # Path configs
    |   data_loader.py # Data loader
    |   evaluation.py # Evaluation metrics
    |   network.py # Neural network design
    |   plotting.py # Plotting functions 
    |   preprocessing.py # Preprocessing functions
```
## Usage 
To train and test the networks, run the command in your terminal:
```python
python main.py
```
*The classifier runs have been commented out for speed and to test the neural network*
To run with tall of the classifiers, uncomment line 25 and comment ("ctrl /") line 26 in main.py:
```python
    # classifiers = get_classifiers() ## UNCOMMENT THIS
    classifiers = {} # COMMENT THIS OUT
```