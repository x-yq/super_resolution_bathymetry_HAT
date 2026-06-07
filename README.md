# Super Resolution of Ocean Imagery for Improving Bathymetry Prediction and Pixel-based Classification



## Getting started

### Cloning the repository
#### 1) Using the cv4rs Server
If you are working on the cv4rs server, you do not need to clone the repository as it is already installed. Starting from the home directory, you can access it via this command:

```
cd ../../faststorage/cv4rs_2024_superpixel/super-resolution-of-ocean-imagery-for-improving-bathymetry-prediction-and-pixel-based-classification/
```

#### 2) Using a Local Machine
Clone the repository via this command:

```
git clone https://git.tu-berlin.de/rsim/cv4rs-2024-summer/super-resolution-of-ocean-imagery-for-improving-bathymetry-prediction-and-pixel-based-classification.git
```

After cloning, navigate to the repository with this command:

```
cd super-resolution-of-ocean-imagery-for-improving-bathymetry-prediction-and-pixel-based-classification/
```

### Prerequisites

Depending on your setup, you may choose to use the existing conda environment on the cv4rs server or a virtual python environment on your local machine.

#### 1) Using the cv4rs Server

If you are working on the cv4rs server, activate the existing conda environment with the following commands:

```
source ../../faststorage/cv4rs_2024_superpixel/miniconda/conda/bin/activate
conda activate sr4o
```

#### 2) Using a Local Machine
Ensure you have Python installed and then set up a virtual environment:

```
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```


### Visualize the Dataset

First, check if the `root_dir` variable inside `visualize.py` is set correctly to the directory containing the dataset. Then create visualizations of the MagicBathyNet dataset with this command:
```
python visualize.py
```

The file `visualization.png` will be created in the current directory:
![Visualization](images/visualization.png)

## Authors
Maximilian Kromer,
Yeqiao Xu,
Niklas Schmolenski

## License
For open source projects, say how it is licensed.
