# ShorNet (code)
## Training, inference and support scripts for [ShorNet](https://huggingface.co/maxhirez/ShorNet)
*(From huggingface model repo)*

ShorNet is a a multi-layer perceptron for predicting two prime factors from large 1024-bit hex numbers. It was conceived as a function approximation of the Shor Algorithm for prime factorization intended for theoretical future quantum computers.

## Background
ShorNet was conceived as a project for learning about neural networks and cryptography and for testing the abilities of the Macintosh Studio M3 Ultra.  It was inspired by the success of Google Deepmind's AlphaFold, which exhibits high effectiveness at predicting protein folding, which was similarly thought only to be solvable by mega- to giga-qubit quantum computers.

### Manifest

-**Bash-support.txt**: Shell scripts for easy paste execution of scripts (actually zsh not bash) 

-**finetune_script_fixed.py**: Continues training of model trained with *pytorch_prime_factorization.py*.

-**model_loader.py**: Tests trained model against examples from dataset

-**prime_factor_inference.py**: Script for using trained model to predict factors of argued 1024bit hex number

-**prime_factor_utils.py**: Tests trained model on examples from dataset and internally generated samples.

-**Primes_set.ipynb**:Jupyter notebook for generating sets of 512bit prime numbers in hex format and cross multiplying them all into .csv files of product "p" and factors "q" and "n", shuffles rows.

-**pytorch_prime_factorization.py**: Main training script

-**requirements.txt**: Conda environment requirements
