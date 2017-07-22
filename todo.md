software:
- write code to get frames from camera along with a bounding box (user defined?)
- write 2048 implementation in python and figure out what gestures to use to control it (left right up down pointing?)
- create a GUI for the game? Electron?

ML:
- define resolution of images, ideally as low as possible to speed up training and evaluation
- get a dataset
    - train
    - validation
    - test 
    - could use the dude's dataset as a starting point

- augment dataset using translations, rotations (, light variation?)
    - include augmentation in training script to reduce the amount of data to store

- find best CNN architecture for task (MNIST CNN architecture apparently)
    - any good heuristics to help determine ideal architecture?
    - dropout? batch normalisation? ReLU vs LReLU vs ELU vs SELU?
    - batch size
    - Adam vs Nesterov-Momentum

- figure out whether preprocessing the images provides a boost in accuracy

- train final network and figure out how to serve it
    - package network into easy to use python class
        - interactive tf session
        - class GestureNetwork(load_path = path)
        - net.predict(thumbnail)

Extra:
- write blog post about it on portfolio

- package the code so that it is easy to use
    - provide source to run the program with clear dependency outline (probably opencv, tensorflow/keras, h5py for the dataset)

    - provide training script that is easy to use with documentation on how to use it
        - training script includes data augmentation routines
        - training script uses tf.queues to make the training more efficient
        - training script uses tf.Summary and checkpointing

    - provide some starter training data (in hdf5 format ?) along with documentation on what it includes
    - provide a pretrained model
    - provide some speed (CPU and GPU) and accuracy benchmarks (with/without augmentation)
    - provide hardware requirements for real-time operation

- provide dockerised version
    - both CPU and GPU versions (with nvidia-docker)
    - create Dockerfile
    - create install_dependencies.sh
        - compile tensorflow from source for advanced instructions and CudNN6?

    - if web version, call the docker container from the interface backend
