
With the prevalence of more technology, children access it way before they are of legal age and with little cognitive development. An alarming problem in this regard is children engaging with predators in online grooming conversations. Besides deep web that used to be a hub forillegal activities including child pornography, main stream digital platforms such as online videogames and chat rooms are currently the more common places where children are present and are easy prey for online sexual predators such as those who are diagnosed with paedophilia; an act of an adult having sexual involvement with a minor, through grooming where the sexual predator tries to form emotion relationship with a minor in order to get her trust and make her engage in sexual activities afterwards. 

## Objectives

Around 60%-80% of female high school students have to face online sexual grooming incidents in their life. In many of these instances, the predators try to mix explicit remarks in the conversation to get a sense of how they are going to proceed with the victim, which can be extracted by ``natural language processing`` (NLP) techniques and employed by ``machine learning`` methods to catch such predators. 

1. [Setup](#1-setup)
2. [Quickstart](#2-quickstart)
3. [Results](#3-results)
4. [Acknowledgement](#4-acknowledgement)

## 1. Setup
You can setup the code in different ways. You can use package managers such as Mamba and Conda to install the packages. If you prefer using docker, clone this project and build the Dockerfile.
First we need the project source code.
```sh
git clone https://github.com/fani-lab/Osprey.git
cd Osprey
```
### 1.1 Using mamba/micromamba/conda
These package managers basically have similar interfaces. You can install which ever you think is appropriate for your environment and system. We prefer using mamba as it is faster than conda and has more features than micromamba.

```sh
mamba create -n osprey-cuda
mamba activate osprey-cuda
mamba env update -n osprey-cuda --file environment.yml
```

### 1.2 Using Docker
In case you need an image of this project to run it on a cloud server, you can simply clone the project and run:
```sh
docker build -t osprey-cuda-image .
```
Please pay attention to `.dockerignore` file. The docker engine currently ignores `data`, `logs`, and `output` directories. If you do not have the datasets, please add them to respective project directory and when ran the container, mount the directories to the container.
## 2. Quickstart
You can use the commandline interface (cli) built in this project to run different scripts or train your models. You can use
```sh
python runner.py command
```
and replace `command` with the desired value. You can also run the following command for more help
```sh
python runner.py --help
```

In order to turn the PAN-12 xml files to csv, use this command:
```sh
python runner.py xml2csv --xml-file /path/to/pan12.xml --predators-file /path/to/predtors-ids.txt
```
`xml2csv` creates the v2 dataset. The old code of creating dataset had some limitations and we reimplemented it and named it v2 dataset.
For creating the conversation dataset, where each record is a whole conversation, run the following command. Remember to put both train.csv and test.csv file under the same directory and pass that directory as `--datasets-path` argument.

```sh
python runner.py create-conversations --datasets-path /path/to/dataset-v2/ --output-path /path/to/dataset-v2/conversation/
```

You can also create toy set for conversation dataset using the following command. The ratio value here specifies the ratio of number of original dataset records to that of toy dataset.
```sh
python runner.py create-toy-conversation --train-path /path/to/dataset-v2/conversation/train.csv --test-path /path/to/dataset-v2/conversation/test.csv --ratio 0.1
```

You can define your configurations for sessions and models under the path `settings/settings.py`. There are samples under the same file in `datasets` and `sessions` dicts.
After specifying the sessions configurations according to your need, you can use the following command to run all of `sessions`.
```sh
python runner.py train --log
```

## 3. Results

We report the predictions, evaluation metrics on each test instance, and average on all test instances in **. 

TODO: Put result figures and explain them.

## 4. Acknowledgement
