# Patent Image Retrieval with Graph Alignment and Hyperbolic Projection

This repository contains the code and experiments for the thesis project **"Patent Image Retrieval with Graph Alignment and Hyperbolic Projection"**.

## 📄 Overview

This project explores multimodal patent image retrieval by combining:
- **Graph alignment techniques** for structured metadata,
- **Hyperbolic projection** for better representation of hierarchical relations.

It supports multiple retrieval methods via a unified training and evaluation pipeline.

---

## 📦 Data

Download the patent dataset from the following source:  
👉 https://www.nature.com/articles/s41597-023-02653-7

Once downloaded, organize it according to the expected format inside a `data/` directory.

---

## 🛠 Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Activate the environment

```bash
make activate
```

### 3. Navigate to the source folder

```bash
cd src
```

---

## 🚀 Running the Project

To train a method of your choice:

```bash
python train.py --method <method_name>
```

Replace `<method_name>` with one of the available approaches, such as:
- `clip`
- `clip_ft`
- `clip_graph`
- `clip_hyperbolic`
- `clip_graph_hyp`  
(Refer to the `train.py` file for all options.)

---

## 📊 Graph-Based Retrieval & Testing

Graph-specific methods and retrieval testing are implemented in:

```bash
retrieval.ipynb
```

Use this notebook to:
- Run evaluations on trained models
- Visualize retrieval results
- Apply graph alignment-based retrieval

---

## 📁 Project Structure

```bash
├── data/               # Patent dataset (images + metadata)
├── src/                # Training and model code
│   ├── train.py        # Main training script
│   └── ...             # Supporting modules
├── retrieval.ipynb     # Graph-based retrieval and testing notebook
├── requirements.txt    # Python dependencies
├── Makefile            # Environment activation command
└── README.md           # This file
```

---

## 📚 Citation

If you use this work or dataset in your research, please cite the original dataset publication:

> **Chen et al.**  
> *Open-access multimodal dataset for patent analysis and retrieval*.  
> Nature Scientific Data, 2023. [Link](https://www.nature.com/articles/s41597-023-02653-7)

---

## 🧑‍💻 Author

**[Your Name]**  
Thesis project @ [Your University / Lab]  
2025
