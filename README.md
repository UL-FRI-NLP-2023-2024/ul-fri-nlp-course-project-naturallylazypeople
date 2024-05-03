# Natural language processing course 2023/24: `Parameter-Efficient Fine-Tuning of Language Models`

## Objective of Project

This project is conducted within the course "NLP" at University of Ljubljana (FRI). The objective is to implement and compare different PEFT (Parameter-Efficient Fine-Tuning) Methods for differenft NLP tasks. In our analysis, we will compare the following three methods:
1. LoRA
2. Soft Prompting
3. BitFit.

## Overview of the Tasks

We will compare the methods based on the following benchmarks and their respective tasks:
| Benchmark                              | NLP Task                             |
|----------------------------------------|--------------------------------------|
| CommonsenseQA                          | Commonsense Reasoning                |
| CoNLL-2012                             | Coreference Resolution               |
| XSum                                   | Text Summarization                   |
| SST5                                   | Sentiment Analysis                   |
| Slovene SuperGLUE                      | Slovene BoolQ (Boolean Questions)    |


## Project Structure

The project structure is organized as follows:

- `data/`: Contains the data.
- `report/`: Contains the latex report of our results.
- `src/dataset_handler`: These dataset handlers are used for reading and processing the data according to their objectives.
- `src/evaluator`: Trains and evaluates the different models.
- `src/trainers`: Every task has its own trainer specified in this folder.
- `src/utils`: Contains helper functions.
- `main.py`: Main script that runs the training and evaluation process.
- `output/`: Contains the output (trained models and metrics).

## Running
```bash
cd src
sbatch run.sh
```

## Authors
- Ondra Komin
- Andrej Susnik
- Eileen Vu