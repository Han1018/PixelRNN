# PixelCNN

PyTorch streamlized implementation of PixelCNN from [Pixel Recurrent Neural Networks](http://arxiv.org/abs/1601.06759)

so far is the pixelCNN, i will change the code to pixelRNN
## Todo list:
- [x] streamlized the old code
- [x] apply the old pixelCNN to mnist first to make sure it will work successfully
- [ ] change the pixelCNN to pixelRNN

## Repo env
- cuda 12.1
- Python 3.x
- Pytorch


## Installation
1. Clone the repository:
    ```bash
    $ git clone https://github.com/han1018/PixelCNN.git
    $ cd PixelCNN
    ```

2. Create a virtual environment using conda:
    ```bash
    $ conda env create -f environment.yml
    ```

3. Activate the virtual environment:
    ```bash
    $ conda activate pixelcnn
    ```

## Training
To train the model on MNIST or CIFAR-10, simply run the training script as shown below. Modify the training scripts as needed to adjust hyperparameters or add functionality.

-  MNIST
    ```bash
    $ python3 train_mnist.py
    ```

-  CIFAR-10
    ```bash
    $ python3 train_cifar10.py
    ```

## Contributing
Feel free to submit issues or pull requests if you find any bugs or have suggestions for improvements.