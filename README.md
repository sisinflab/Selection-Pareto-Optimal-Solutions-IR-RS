# Post-hoc Selection of Pareto-Optimal Solutions in Search and Recommendation
This repository contains the source codes and datasets of the paper _Post-hoc Selection of Pareto-Optimal Solutions in Search and Recommendation_ submitted to 
The 32nd ACM International Conference on Information and Knowledge Management (CIKM 2023).

### Requirements
To run these codes, make sure to have a Python `3.8.0` or later version installed on your device. Then, you may create the virtual environment and install the libraries included in the `requirements.txt` file as follows:

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
### Data
In the folder `data`, you can find the data used in our work. 
 - `amazon_music` and `goodreads` folders contain the split version of their corresponding dataset. These datasets are useful to compute the tailored utopia points for the recommendation scenarios. You may compute them by running the following code:
    
```
$ python -u user_popularity.py
```
- The files `EASER_amazonmusic.tsv`, `EASER_goodreads.tsv`, `nn.tsv`, and `trees.tsv` report the aggregated performance values of the several configurations models trained.
- In the folder `population`, you may find the performance values of such models at user/query level for each dataset.

### IR Scenario
Here, we explain how to reproduce the results regarding the Information Retrieval scenario. 
- The script `ir_1.py` reproduces the results concerning the case involving two objectives (accuracy and time per docs). Note you should modify `line 12` of this script at your convenience to run the codes for the LambdaMART or Neural Networks models. To run it, you should run the following command:
```
$ python -u ir_1.py
```
- The script `ir_2.py` reproduces the results concerning the case involving three objectives (accuracy, time per docs, and energy per docs) for the LambdaMART models. To run it, you should run the following command:
```
$ python -u ir_2.py
```
The complete results are reported in the following tables.
##### LambdaMART
![alt text](https://github.com/sisinflab/Selection-Pareto-Optimal-Solutions-IR-RS/blob/main/LambdaMART.png?raw=true)

##### Neural Networks
![alt text](https://github.com/sisinflab/Selection-Pareto-Optimal-Solutions-IR-RS/blob/main/NeuralNetworks.png?raw=true)

### RS Scenario
Here, we explain how to reproduce the results regarding the Recommender Systems scenario. 
- The script `rs_1.py` reproduces the results concerning the Goodreads dataset. To run it, you should run the following command:
```
$ python -u rs_1.py
```
- The script `rs_2.py` reproduces the results concerning the Amazon Music dataset. To run it, you should run the following command:
```
$ python -u rs_2.py
```
The complete results are reported in the following table.
##### EASER
![alt text](https://github.com/sisinflab/Selection-Pareto-Optimal-Solutions-IR-RS/blob/main/EASER.png?raw=true)

## EASER models training
To train the 48 EASER model configurations, we exploited an ad-hoc version of [Elliot](https://elliot.readthedocs.io/en/latest/), which is an open-source recommendation framework. Please, refer to the official documentation of Elliot for more details about the framework. You can download it at this [link](https://drive.google.com/file/d/13a35C1CxXd4jx8oWvYfpvbxX6jJ2h67s/view?usp=sharing). In the folder`config_files`, you can find the configuration files used to train the models.
To run these codes, make sure to have a Python `3.8.0` or later version installed on your device. Firstly, download the zip file from the link provided before. Then, unzip the file. In the project's folder,  you may create the virtual environment with the requirements files we included as follows:

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

To train the models, run the following commands:

```
$ python -u start_experiments.py
```

You will find the results in the folder `results` for each dataset.

