
With the prevalence of more technology, children access it way before they are of legal age and with little cognitive development. An alarming problem in this regard is children engaging with predators in online grooming conversations. Besides deep web that used to be a hub forillegal activities including child pornography, main stream digital platforms such as online videogames and chat rooms are currently the more common places where children are present and are easy prey for online sexual predators such as those who are diagnosed with paedophilia; an act of an adult having sexual involvement with a minor, through grooming where the sexual predator tries to form emotion relationship with a minor in order to get her trust and make her engage in sexual activities afterwards. 

## Objectives

Around 60%-80% of female high school students have to face online sexual grooming incidents in their life. In many of these instances, the predators try to mix explicit remarks in the conversation to get a sense of how they are going to proceed with the victim, which can be extracted by ``natural language processing`` (NLP) techniques and employed by ``machine learning`` methods to catch such predators. 

1. [Setup](#1-setup)
2. [Quickstart](#2-quickstart)
3. [Results](#3-results)
4. [Acknowledgement](#4-acknowledgement)

## 1. Setup

You need to have ``Python >= 3.8`` and install the following main packages, among others listed in [``requirements.txt``](requirements.txt):
```

```
By ``pip``, clone the codebase and install the required packages:
```sh
git clone https://github.com/Fani-Lab/online_predatory_conversation_detection
cd online_predatory_conversation_detection
pip install -r requirements.txt
```
By [``conda``](https://www.anaconda.com/products/individual):

```sh
git clone https://github.com/Fani-Lab/online_predatory_conversation_detection
cd online_predatory_conversation_detection
conda env create -f environment.yml
conda activate online_predatory_conversation_detection
```


## 2. Quickstart

```sh
cd src
python main.py 
```

The above run, loads and preprocesses ** followed by 5-fold train-evaluation on a training split and final test on test set for **.

## 3. Results

We report the predictions, evaluation metrics on each test instance, and average on all test instances in **. 

TODO: Put result figures and explain them.

## 4. Acknowledgement
