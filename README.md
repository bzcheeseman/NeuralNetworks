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
* COMMENT - it works NOW COMMENT IT
* Implement regularization
* Read from/Write to file
* Other, fancier networks!



Dependencies:
```
cmake
Eigen
gflags
```

On Mac:
```
brew install cmake
brew install Eigen
brew install gflags

```

If homebrew is not installed, follow the official [Eigen instructions](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download),
the official [cmake instructions](https://cmake.org/install/),
and the official [gflags instructions](https://gflags.github.io/gflags/) to install
the required dependencies.
  For Eigen, if it's installed in a place other than `/usr/local/include/eigen3` then change the include_directories macro in the `CMakeLists.txt` file (line 4)
