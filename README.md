# TradingView Chart Pattern Classifier

This project implements a deep learning model to classify common chart patterns from TradingView candlestick chart images. It's a computer vision application focused on recognizing visual trends in financial charts.

## ⚠️ Disclaimer

This project is for **educational and experimental purposes only** and should **NOT** be used for actual financial trading decisions. Financial markets are complex and highly unpredictable. Any use of this software for real trading is at your own risk.

## Project Overview

The goal is to classify images of candlestick charts into predefined patterns such as "Uptrend," "Downtrend," "Sideways/Range," or specific buy/sell signals.

## Features

* **Image Classification:** Utilizes Convolutional Neural Networks (CNNs) for chart pattern recognition.
* **Transfer Learning:** Employs a pre-trained ResNet backbone for efficient feature extraction.
* **PyTorch Framework:** Built using the PyTorch deep learning library.
* **Custom Dataset Handling:** Includes a custom `ChartDataset` for loading and transforming image data from a structured directory.
* **MPS Support:** Configured to leverage Apple Silicon's Metal Performance Shaders (MPS) for GPU acceleration on compatible Macs.

## Dataset

The dataset consists of `102` manually collected screenshots of TradingView charts, categorized into various chart patterns. The dataset is organized into `train`, `val`, and `test` directories, with subdirectories for each pattern class.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd trading_project
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows: .\venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    * *Note for Apple Silicon (M1/M2/M3):* Ensure your PyTorch installation supports MPS. The `pip install torch torchvision torchaudio` command (when run on macOS) should typically handle this.

## Usage

1.  **Prepare your dataset:**
    * Create a `dataset/trading_chart_dataset/` directory in the project root.
    * Inside it, create `train/`, `val/`, and `test/` folders.
    * Within each of these, create subfolders for your pattern classes (e.g., `uptrend/`, `downtrend/`, `sideways/`).
    * Place your collected chart image screenshots into the respective class folders.

2.  **Run the training and evaluation script:**
    ```bash
    python main.py
    ```
    The script will train the model, validate its performance, and evaluate it on the test set, displaying accuracy, F1-score, and a confusion matrix. A trained model will be saved as `trained_model.pth`.

## Project Structure