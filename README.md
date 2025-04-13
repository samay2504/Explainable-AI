# Explainable-AI: Extended Probabilistic Label Trees (EPLT)

[![Python build](https://github.com/mwydmuch/napkinXC/actions/workflows/python-test-build.yml/badge.svg)](https://github.com/mwydmuch/napkinXC/actions/workflows/python-test-build.yml)
[![Documentation Status](https://readthedocs.org/projects/napkinxc/badge/?version=latest)](https://napkinxc.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

The **Explainable-AI** repository extends the [napkinXC](https://github.com/mwydmuch/napkinXC) framework to experiment with Extended Probabilistic Label Trees (EPLT) and add explainability features. In this project we introduce adversarial training, extended evaluation routines, and detailed visualizations to better understand and interpret extreme multi-label classification models.

We have made several modifications over the original napkinXC codebase:
- Added adversarial training routines and evaluation code in `EPLT_expirement.py`
- Included new Jupyter notebooks (e.g., `Extended_Probabilistic_Label_Trees.ipynb`) with detailed results for datasets such as **eurlex-4k** and **mediumill**
- Adjusted certain components to support explainability and improved evaluation visualizations

For full project details and source code, please visit our GitHub repository at: [https://github.com/samay2504/Explainable-AI](https://github.com/samay2504/Explainable-AI)

---

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Experiment Script](#running-the-experiment-script)
  - [Working with Notebooks](#working-with-notebooks)
- [Project Details](#project-details)
  - [Extended PLT Model](#extended-plt-model)
  - [Adversarial Training & Evaluation](#adversarial-training--evaluation)
- [Reproducing Results](#reproducing-results)
- [Acknowledgments and References](#acknowledgments-and-references)
- [License](#license)

---

## Overview

The Explainable-AI project builds on napkinXC with additional experimental work in extreme multi-label classification:
- **Adversarial Training:** Enhances the baseline Probabilistic Label Trees (PLTs) by training with synthetic noisy inputs.
- **Explainability:** Implements functions to extract and visualize the top influential features for predictions.
- **Robust Evaluation:** Extends testing with noise variations to evaluate metrics such as Precision@k, nDCG@k, and PSDCG@k, along with prediction timing.
- **Visualizations:** Produces multiple plots that show how noise affects performance, aiding in interpretation and analysis.

---

## Directory Structure

The repository is organized as follows:

```
mwydmuch-napkinxc/
├── README.md
├── Extended_Probabilistic_Label_Trees.ipynb       # Jupyter notebook demonstrating EPLT results
├── EPLT_expirement.py                              # Main experiment script with argparse
├── CMakeLists.txt
├── format_code.sh
├── LICENSE.md
├── MANIFEST.in
├── pyproject.toml
├── setup.py
├── .clang-format
├── .readthedocs.yaml
├── docs/                                         # Documentation files and static assets
├── experiments/                                  # Scripts and experiments to reproduce results
├── pybind11/                                   # External dependency for C++ binding to Python
├── python/                                     # Python-specific source code and examples
├── src/                                        # C++ source code for core components
└── .github/                                    # GitHub workflows and CI/CD configuration
```

---

## Installation

### Prerequisites

- **Python 3.9+**
- A modern **C++17** compiler
- **CMake**
- **Git**

### Clone the Repository

```bash
git clone https://github.com/samay2504/Explainable-AI.git
cd mwydmuch-napkinxc
```

### Set Up the Environment

Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### Install Dependencies

Install required Python packages using:

```bash
pip install -r docs/requirements.txt
```

If necessary, install napkinXC from source (or using the GitHub link):

```bash
pip install git+https://github.com/mwydmuch/napkinXC.git
```

Then install the package:

```bash
python setup.py install
```

---

## Usage

### Running the Experiment Script

The core experiment for training and evaluation is implemented in `EPLT_expirement.py`. It uses argument parsing to allow flexible configuration:

```bash
python EPLT_expirement.py --dataset_name eurlex-4k --noise_level 0.05 --augment_ratio 1.0
```

This command will:
- Load the specified dataset (e.g., `eurlex-4k`)
- Train a normal model along with an adversarially trained model (using user-provided noise parameters)
- Perform evaluation on the test set, printing performance metrics (Precision@k, nDCG@k, PSDCG@k) and generating various plots for analysis

### Working with Notebooks

For interactive analysis and visualization, open the provided Jupyter Notebook:

```bash
jupyter notebook Extended_Probabilistic_Label_Trees.ipynb
```

This notebook details the experiments, showcases results on datasets like **eurlex-4k** and **mediumill**, and includes code for feature importance and noise impact visualization.

---

## Project Details

### Extended PLT Model

The Extended PLT model inherits from the napkinXC **PLT** class and adds:
- **Adversarial Training:** Augmenting training data with controlled noise (`noise_level` and `augment_ratio` parameters).
- **Explainability Functions:** Providing explanations for individual predictions by identifying top contributing features.
- **Robust Evaluation:** Testing the sensitivity of predictions under various noise levels, reporting metrics including prediction time.

### Adversarial Training & Evaluation

Key aspects introduced in this project:
- **Adversarial Training:** The script `EPLT_expirement.py` calls a specialized training routine to fit adversarial examples.
- **Evaluation Metrics:** Beyond standard accuracy metrics, our evaluation includes:
  - **Precision@k**
  - **nDCG@k**
  - **PSDCG@k**
  - **Prediction Times**
- **Visualization:** Several functions plot metric trends vs. noise levels and highlight feature importance for individual test samples.

---

## Reproducing Results

1. **Data Preparation:**  
   Utilize napkinXC’s built-in functions (or provided scripts in the `experiments/` folder) to download or preprocess datasets like `eurlex-4k` and `mediumill`.

2. **Run the Experiment:**  
   Execute the experiment script with the desired parameters:
   ```bash
   python EPLT_expirement.py --dataset_name eurlex-4k --noise_level 0.05 --augment_ratio 1.0
   ```
3. **Review Notebooks:**  
   Open `Extended_Probabilistic_Label_Trees.ipynb` to explore detailed experimental results.

---

## Acknowledgments and References

The foundation for this repository is built on the napkinXC framework by [Marek Wydmuch](https://github.com/mwydmuch/napkinXC). Additional references used in this project include:

- **Probabilistic Label Trees for Extreme Multi-label Classification**  
  [arXiv:2009.11218](https://arxiv.org/abs/2009.11218)
- **Online Probabilistic Label Trees**  
  [Proceedings of the Machine Learning Research (MLR)](http://proceedings.mlr.press/v130/jasinska-kobus21a.html)
- **Propensity-scored Probabilistic Label Trees**  
  [ACM Digital Library](https://dl.acm.org/doi/10.1145/3404835.3463084)
- **Efficient Algorithms for Set-Valued Prediction in Multi-Class Classification**  
  [Springer Link](https://link.springer.com/article/10.1007/s10618-021-00751-x)

---

## Authors
```
Atharva Date @adiitj
Samay Mehar @samay2504
```

---

Feel free to contribute, report issues, or share enhancements. We welcome feedback and contributions to further improve the explainability and robustness of EPLT.

---

This README should serve as a thorough guide for setting up, running, and understanding the Extended Probabilistic Label Trees experiments on extreme multi-label classification datasets. Enjoy exploring Explainable-AI!
