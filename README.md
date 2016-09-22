After downloading and unzipping (Make sure to check dependencies!), usage:

```
mkdir build && cd build
cmake ..
make

//And now run the code!  This will train the network on the iris dataset and output results!
./NeuralNetworks
```

In order to play around with parameters, look in the `main.cpp` file,
should be self explanatory.  Change parameters and sizing as desired.

TO DO:
* Finish commenting
* Read from file
* Other, fancier networks!

This code for now implements a basic feed forward neural network.
This network is parallelized using OpenMP to speed up training.

It contains implementations for the AdaDelta method of backpropagation, as well as 
Momentum Gradient Descent and standard Stochastic Gradient Descent.

Also has options for dropout layers during training, as well as L1 or L2 regularization,
or no regularization at all.

The network is implemented such that it performs online learning - that is, weights are
updated as training happens.

Working on a CUDA implementation as well as fancier architectures.

To view the documentation with [electron](https://github.com/electron/electron/blob/master/docs/tutorial/quick-start.md), first install electron as detailed in the link and then simply
```
cd docs
electron .
```
and the Doxygen documentation will pop up in its own window! Cool, huh?

Dependencies:
```
cmake
Eigen
gflags
CUDA (>= 7.0) - including cuBLAS especially
CUDA - cuDNN  (>= 5.0)
```

On Mac:
```
brew install cmake
brew install Eigen
brew install gflags
brew cask install cuda
```

Note: To download cuDNN requires a free account for the accelerated computing program at NVIDIA.

cuDNN setup from the TensorFlow [setup page](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#prepare-environment-for-mac-os-x)

---------------------------------------------------------------------------
From TensorFlow setup page:
```
$ sudo mv include/cudnn.h /Developer/NVIDIA/CUDA-7.5/include/
$ sudo mv lib/libcudnn* /Developer/NVIDIA/CUDA-7.5/lib
$ sudo ln -s /Developer/NVIDIA/CUDA-7.5/lib/libcudnn* /usr/local/cuda/lib/
```
To verify the CUDA installation, you can build and run deviceQuery to make sure it passes.
```
$ cp -r /usr/local/cuda/samples ~/cuda-samples
$ pushd ~/cuda-samples
$ make
$ popd
$ ~/cuda-samples/bin/x86_64/darwin/release/deviceQuery
```
---------------------------------------------------------------------------

If homebrew is not installed seriously consider [installing it](http://brew.sh/) (for Macs - it's great, really), otherwise follow the official [Eigen instructions](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download),
the official [cmake instructions](https://cmake.org/install/), the official [gflags instructions](https://gflags.github.io/gflags/), and the official [cuda instructions](https://developer.nvidia.com/cuda-downloads) to install
the required dependencies.
  
For Eigen, if it's installed in a place other than `/usr/local/include/eigen3` (the default location for Homebrew) then change the include_directories macro in the `CMakeLists.txt` file (line 39)
