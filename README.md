This repository contains the code and datasets used in the paper "Large Language Models and Natural Language in Code: Towards More Resilient LLMs".

It contains the following components:

- The datasets, containing the return-type-stripped code and corresponding return types, used to test the LLMs and to train the BERT models.
- Code for evaluating the LLMs is found in llm-return-type-prediction-next-token.ipynb.
- Code for training the BERT models is found in codebert-classifier.py

To run the code, you will need to install the required packages, found in requirements.txt. You can install them using pip:

```bash
pip install -r requirements.txt
```

To run the evaluation code, open the Jupyter notebook `llm-return-type-prediction-next-token.ipynb`, edit the output paths, and run the cells.

To train the BERT models, run the script `codebert-classifier.py`. This script will output the trained models to the `models/` directory.