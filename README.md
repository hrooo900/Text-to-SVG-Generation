# Drawing with LLMs: Fine-Tuning Qwen3-4B using Unsloth

This project demonstrates how to fine-tune the [Qwen3-4B](https://huggingface.co/Qwen/Qwen1.5-4B) large language model using the [Unsloth](https://github.com/unslothai/unsloth) library for the [Drawing with LLMs](https://www.kaggle.com/competitions/drawing-with-llms) competition. The goal is to train a model that generates high-quality SVG code from natural language prompts.

## Project Structure

- `unsloth-qwen3-4b-FINE-TUNING.ipynb`: Main notebook for model fine-tuning and evaluation.
- `Testing-my-model.ipynb`: (Not detailed here) Presumably for testing the trained model.

## Workflow Overview

1. **Environment Setup**
   - Installs required libraries: `torch`, `xformers`, `unsloth`, and `pip3-autoremove`.

2. **Model Loading**
   - Downloads and loads the Qwen3-4B model using Unsloth's `FastLanguageModel`.
   - Configures model parameters such as sequence length and device.

3. **LoRA Configuration**
   - Applies Low-Rank Adaptation (LoRA) to the model for efficient fine-tuning.
   - Sets LoRA parameters like rank, target modules, and dropout.

4. **Data Preparation**
   - Downloads the [SVG Generation Sample Training Data](https://www.kaggle.com/datasets/vinothkumarsekar89/svg-generation-sample-training-data) from Kaggle.
   - Loads and previews the data using pandas.

5. **Prompt Engineering & Visualization**
   - Defines a system prompt for SVG code generation.
   - Implements a function to generate SVG code from descriptions and visualize results.

6. **Dataset Formatting**
   - Formats the dataset into instruction-following prompts (Alpaca style).
   - Converts the pandas DataFrame to a Hugging Face `Dataset` and applies formatting.

7. **Fine-Tuning**
   - Sets up the [TRL SFTTrainer](https://github.com/huggingface/trl) for supervised fine-tuning.
   - Configures training arguments (batch size, epochs, optimizer, etc.).
   - Trains the model on the formatted dataset.

8. **Model Saving & Upload**
   - Merges LoRA weights and saves the full model in FP16 format.
   - Optionally uploads the trained model to [KaggleHub](https://www.kaggle.com/docs/models).

## Usage

1. **Run the Notebook**
   - Open [`unsloth-qwen3-4b-FINE-TUNING.ipynb`](unsloth-qwen3-4b-FINE-TUNING.ipynb) in VS Code or Jupyter.
   - Execute cells sequentially to install dependencies, load data, fine-tune the model, and save/upload the results.

2. **Modify Parameters**
   - Adjust model parameters, LoRA settings, and training arguments as needed for your experiments.

3. **Test the Model**
   - Use the provided `gen` function to generate SVG code from sample descriptions and visualize outputs.

## Requirements

- Python 3.8+
- CUDA-enabled GPU (for efficient training)
- Libraries: `torch`, `transformers`, `unsloth`, `trl`, `datasets`, `kagglehub`, `pandas`, `xformers`, `IPython`

## References

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Qwen3-4B Model](https://huggingface.co/Qwen/Qwen1.5-4B)
- [Drawing with LLMs Competition](https://www.kaggle.com/competitions/drawing-with-llms)
- [SVG Generation Sample Training Data](https://www.kaggle.com/datasets/vinothkumarsekar89/svg-generation-sample-training-data)

---

**Author:** Hritwij Kamble
**Date:** 2025-05-05
