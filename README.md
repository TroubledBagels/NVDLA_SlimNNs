
# NVDLA_SlimNNs
This is the implementation of a dynamic neural network, specifically slimmable neural networks (SlimNNs) atop of the software stack of the NVIDIA Deep Learning Accelerator. This has been done in two ways, with a single User Mode Driver (UMD) and with multiple UMDs. Below are instructions for usage.

N.B. When cloning, for the full Runtime implementation, make sure you do a recursive update for submodules:
````
git submodule update --init --recursive
````

## Workflow
### Model Creation
The file ````AlexSNN.py```` gives a structure for an AlexNet adaptation for CIFAR10 (upscaled to 64x64). You can create a model and then use code similar to that in ````make_alex_model.py```` to train and run the model, which should then be saved into the snn_models folder.

Usage: ````python3 make_alex_model.py <int: parts>````
N.B. Partitions (i.e. ````width_mults```` are currently manually defined in the ````AlexSNN```` class).

The file ````generate_graphs.py```` creates graphs for mapping out a few factors for different thresholds, i.e. the confidence required for an accepted output (and exports to model_images). The created graphs include:

- Accuracy against Threshold
- Time against Threshold (N.B. this isn't a normalised time, it measures the execution time for your machine)
- Confidence against Threshold (min, Q1, median, Q2, max, and mean)
- A 3D graph showing how often each width is the final width used (i.e. how many prediction have enough confidence to pass on each width)

Usage: ````python3 generate_graphs.py <model_name>````
N.B. The model name shouldn't include the file path, unless it is stored in nested directories within ````snn_models````, in which case only include the nested directories, e.g. ````alex_models/model_2_part```` will load the file from ````snn_models/alex_models/model_2_part.py````.

If you're creating your own model for this, you should  refer to the ONNC release page to see supported layers:
https://github.com/ONNC/onnc/releases

### Model Compilation
Compilation has 3 steps:
1. Separate partitions
2. Export said partitions to ONNX
3. Use the ONNC compiler to create the ````.nvdla```` file

#### Separating Partitions and Exporting to ONNX
This can be done in a single file, or separately depending on what you want to do. The file ````output_each_partition.py```` allows for both in one go, and is the most efficient way to do this.

Usage: ````python3 output_each_partition.py <model_name>````
N.B. Uses the same model name logic as ````generate_graphs.py```` (see above)

When creating your own model, you will need to define your own "static form" of the SlimNN you have created. This is a basic form of the network, without any of the dynamic logic included.

This script outputs into two files. ONNX files get exported into the ````onnx```` directory within ````partitioned_networks````, and PTH files into the ````pth```` directory. This allows for easy access of all forms of then network.

As the ONNC compiler is built with ONNX opset version 8 (with a few edits) there's a separate file for exporting and verifying a ````.pth```` to an ````.onnx````. This is ````pth_to_onnx.py````. This script takes a single ````.pth```` model from ````snn_models```` and exports it into the ````onnx_models```` directory. It was created while attempting to export the original dynamic AlexSNN model into ONNX, which did work, however ONNC is yet to support the "slice" operation, and so it cannot be compiled.

Usage: ````python3 pth_to_onnx.py <model_name>````

#### Compilation with ONNC
As can be seen, there is no ONNC executable or related file within the repository. That is because, the most efficient way to compile a new model is through the docker.
````
docker pull onnc/onnc-community
docker run -ti --rm onnc/onnc-community /bin/bash
````
This will boot you into the onnc-community docker, and puts you into the correct directory. From there you can find the ONNC compiler in ````<insert path>/onnc````. You can then use this as follows (I recommend copying the model you want to compile into the ````/models```` directory from the root).
````
./<insert_path>/onnc -mquadruple nvdla /models/<model_name>.onnx
````
For some (unknown) reason, a few of the provided models don't work with the virtual platform, most notably ````bvlc_alexnet````.

This will now give you (in your directory) a file called ````out.nvdla````. Copy this out of the docker and you have your compiled NVDLA loadable.

### Model Runtime
#### General Usage
The first step for executing the model is using the NVDLA virtual platform (specifically the ONNC version, which has a few edits). Again, there is a set of dockers for this.
````
docker pull onnc/vp
docker run -it -v /home:/home onnc/vp
````
This will load you into the ONNC virtual platform (VP) docker. From here (if you are not already in the directory), you should navigate to ````/usr/local/nvdla````. This is the main directory for running the virtual platform. From here, you should run the following commands:
````
aarch64_toplevel -c aarch64_nvdla.lua  % Launches the VP emulator
	Username: root
	Password: nvdla
mount -t 9p -o trans=virtio r /mnt  % Mounts the drive
cd /mnt
./init_dla.sh  % loads the nvdla kernel module
````
From here, you are ready to use the runtime. First thing, is to get an image and the loadable. Download a related image and make sure it is the correct scale (if it has been transformed for your network). Copy that into the correct folder:
````
docker ps  % This will list the active, running containers - find the one that says onnc/vp
docker cp <your_dir>/image.jpg <vp_container_id>:/usr/local/nvdla/image.jpg
docker cp <your_dir>/loadable.nvdla <vp_container_id>:/usr/local/nvdla/loadable.nvdla
````
Now that the required folders are in the directory, you can run an inference:
````
./nvdla_runtime --loadable loadable.nvdla --image image.jpg --rawdump
````
This will run and create an ````output.dimg```` file. The argument ````--rawdump```` makes this human-readable. These are the results of the inference.

#### Custom Runtimes - General Information
When creating a custom runtime, there are a few steps to go through. Within the ````Runtime```` directory, there is an edited version of the original NVDLA ````sw```` repository, containing two example runtimes within 
````Runtime/sw/umd/tests/runtime````.

These are split into the single-UMD and multi-UMD versions. At each point there is one file that doesn't have an identifier and one that does, e.g. ````RuntimeTest.cpp```` and ````RuntimeTest_multi.cpp````. Currently, you cannot compile from any other name apart from the original (i.e. ````RuntimeTest.cpp````), so each time you want a new compilation, change the names of the files you no longer want to compile to something else, e.g. ````RuntimeTest_single.cpp````, and those that you do want to compile to the originals. The files that will need a change are:

- ````RuntimeTest.cpp````
- ````RuntimeTest.h````
- ````main.cpp````
- ````main.h````
- ````Server.cpp````
- ````Server.h````

These are the only files that need name changes in the ````sw```` repository, however a few others have been edited. 

- ````TestUtils.cpp```` has been edited so that the ````--normalize```` argument accepts 3 values, one for each channel, rather than just 1
- ````DlaImage.cpp```` has been changed for compatibility
- ````DlaImageUtils.cpp```` has been changed for compatibility

[Explanation about AppArgs, etc.]

#### Custom Runtimes - Compilation
To compile the runtime, you have two options.
1. Set up an environment that will be able to perfectly execute the  code (long process - unrecommended)
2. Use another Docker

For the docker, I used the 16:04
