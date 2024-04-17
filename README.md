# Acquiring Complex Concepts with Comparative Learning
Official codebase for the workshop poster #10 from D. Calanzone, F. Merlo, presented at [CORE 2024 Workshop](https://eventum.upf.edu/111520/detail/core-project-2024-workshop-n-information-theoretic-perspectives-on-referring-expression-choice.html) in Universitat Pompeu Fabra, Barcelona.

📌 [Paper is accessible here!](https://halixness.github.io/assets/pdf/complex_comparative_learning.pdf)

## Setup
You can install the environment `comparative_learning` with Anaconda through the provided file `environment.yml`:
```
conda env create -f environment.yml
```
As we rely on [OpenAI CLIP](https://github.com/openai/CLIP), you will have to install the package from the repo:
```
pip install git+https://github.com/openai/CLIP.git
```

## Experiments
Our repo is based on the work from [Bao et al. 2023](https://github.com/sled-group/Comparative-Learning), where you can find access to the SOLA dataset.

You can test the **Hyper-Network** by running:
```
python train_hypernetwork.py -i path_to_SOLA
```
Further options, such as `buffer_size` for experience replay, are available by passing the flag `-h`.

To test **Modular Skill Sharing** instead, run:
```
python train_mss.py -i path_to_modified_SOLA --accumulation_steps 128 --lr 0.0003 --n_concepts 3 --offline True --epochs 1
```
The modified version of SOLA for Logical Pattern Recognition is accessible at here (TBD).

**Note:** the training scripts support weights and biases (experimental). For modular skill sharing, we performed hyperparameter search by running a sweep (`sweep.yml`). The script `run_agents.sh` allows to easily run multiple agents for a single sweep on multiple GPUs.

## Credits
- [McGill-NLP/polytropon](https://github.com/McGill-NLP/polytropon): the official implementation for modular skills sharing that we re-adopted.
- [sled-group/Comparative-Learning](https://github.com/sled-group/Comparative-Learning): for the baseline model codebase and the experimental setup we built on.
