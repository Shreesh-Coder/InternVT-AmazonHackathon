# InternVT - Amazon Hackathon

Welcome to the **InternVT** repository, created for the **Amazon Hackathon**. This project focuses on leveraging vision-language models to address challenges related to multi-modal data, including images and text, with a special focus on grounding, detection, text extraction tasks, and model fine-tuning.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Fine-tuning and Inference](#fine-tuning-and-inference)
- [License](#license)
- [Citations](#citations)

## Introduction

This project was developed as part of the **Amazon Hackathon**, and it explores the integration of vision and language models to create solutions capable of interpreting multi-modal inputs. Using the **InternVT** framework, we focus on tasks such as:

- Grounding and detection of entities in images.
- Multi-modal understanding of image-text pairs.
- Text extraction from images (e.g., identifying and extracting quantities like "250 grams" from images).
- Fine-tuning vision-language models on custom datasets for specific tasks.

## Project Structure

```
InternVT-AmazonHackathon/
│
├── dataset/              # Contains train and test CSV files for model evaluation
│   ├── train.csv         # Training dataset
│   └── test.csv          # Testing dataset
│
├── Final internVIT.ipynb  # Main notebook for running the final model experiments, including fine-tuning
├── test.ipynb            # Notebook for running tests and evaluations
├── Untitled.ipynb        # A placeholder or scratch notebook
├── README.md             # Project documentation (this file)
```

## Installation

For installation instructions and dependencies, please check the [official InternVL GitHub repository](https://github.com/OpenGVLab/InternVL).

## Usage

To start running experiments, open the `Final internVIT.ipynb` notebook and run the cells step by step. You can also use the `test.ipynb` notebook for testing purposes.

## Dataset

The dataset used in this project includes:

- **train.csv**: This file contains the training dataset for fine-tuning the model.
- **test.csv**: This file contains the test dataset used to evaluate the fine-tuned model’s performance.

Both files should be placed in the `dataset/` directory before running the notebooks.

## Fine-tuning and Inference

### Fine-tuning

The notebook `Final internVIT.ipynb` includes code for **fine-tuning a pre-trained model** on our custom dataset (train.csv). The fine-tuning process adjusts the pre-trained model to improve its performance on our specific image-text task, such as extracting quantities from images or grounding visual entities.

The fine-tuning process involves:

1. Loading the pre-trained model and tokenizer from the `transformers` library.
2. Defining a training loop to fine-tune the model on our custom dataset.
3. Adjusting the learning rate, batch size, and other hyperparameters for optimal performance.

You can access the **8B InternVL raw fine-tuned checkpoint** used during this process [here on Hugging Face](https://huggingface.co/Shreesh-Coder/interntViT-raw-finetune-Amazon).

### Inference

After fine-tuning, the notebook also includes **code for running inference** using the fine-tuned model on a set of images and questions. This allows us to:

- Generate predictions on the test set (test.csv).
- Extract text (e.g., quantities like "250 grams") from images and compare the results with the ground truth.
- Evaluate the accuracy and performance of the fine-tuned model.

You can access the **final fine-tuned model** ready for inference [here on Hugging Face](https://huggingface.co/Shreesh-Coder/internvit).

### Summary of Code Flow

1. **Fine-tuning**: Fine-tune the pre-trained model on the `train.csv` dataset.
2. **Inference**: Run inference using the fine-tuned model on the `test.csv` dataset.
3. **Compare predictions**: Evaluate the accuracy of the model by comparing its predictions with the ground truth.

### Conclusion

The notebook provides a complete workflow for fine-tuning a vision-language model on a custom dataset and running inference to evaluate its performance on specific tasks such as text extraction and visual grounding.

## License

This project is licensed under the MIT License.

## Citations

This project is inspired by and makes use of open-source models and research from the **InternVL** framework. Please cite the following works if you use this code or dataset:

```bibtex
@article{chen2023internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2312.14238},
  year={2023}
}

@article{chen2024far,
  title={How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
```

For more details on the **InternVL** framework, check out the official [InternVL GitHub Repository](https://github.com/OpenGVLab/InternVL).
