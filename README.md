
# NVDLA_SlimNNs
This is the implementation of a dynamic neural network, specifically slimmable neural networks (SlimNNs) atop of the software stack of the NVIDIA Deep Learning Accelerator. This has been done in two ways, with a single User Mode Driver (UMD) and with multiple UMDs. Below are instructions for usage.

N.B. When cloning, for the full Runtime implementation, make sure you do a recursive update for submodules:
````
git submodule update --init --recursive
````
## Quick "Good to Knows"
1. To exit the NVDLA virtual platform, use ````Ctrl + A```` and then ```X```
2. To exit any docker, just type ````exit```` or press ````Ctrl + D````
3. To copy files into a docker, use ````docker cp <your_dir> <container_id>:<dir>```` (and vice versa)
4. To list running dockers, use ````docker ps````
5. To list all dockers, including the shutdown ones, ````docker ps -a````
6. Don't move the Python files around much, there's quite a few relative paths

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

#### Custom Runtimes - Creation
Couple things to help with this that I couldn't really find elsewhere and had to look for in the code.
There are two main data structures within the runtime. 

The first is ````TestAppArgs````. This structure contains the arguments of the runtime (which can be roughly equated to the UMD), and can be mostly ignored in my personal experience. 

In the multi-UMD runtime, I instantiated multiple of these. It could likely have been done with a slightly smaller memory footprint had I created multiple, but with the current setup of the pre-existing functions within the runtime, it was simpler to keep it the same. Therefore, I created a vector of these, one for each UMD used.
For future work, it may be worth condensing this vector into a single instance of the structure, to do so you would need to vectorise a couple of the stored variables, mainly ````loadableName````, and change the logic of the program to support it.

In the single-UMD runtime, I edited the structure slightly, to contain a vector of loadable names, rather than just storing a single one and passing it along to functions like I had done previously (new_sw commit 8165ca7 and before). This reduced the file sizes slightly and also the memory footprint slightly.

The second data structure is ````TestInfo````, and it is holds the information about the current "Test". Each test comprises loading the image, loading the loadable, creating input and output buffers, and then submitting the task to the accelerator.
The structure of a basic runtime can be put into steps as follows:
1. Parse (and validate) arguments
2. Perform test setup:
    - Check that the image path is valid (not necessarily correct, but valid)
    - Do a little file work
3. Create the runtime instance
4. Read the loadable
5. Load the loadable
6. Initialise the emulator
7. Run the test
    - Validate runtime object
    - Create the input and output buffers
    - Submit the task
    - Output the buffer
8. Clean up

My runtimes essentially follow this structure, with a slight difference as to when and where things are loaded. For example, the multi-UMD runtime loads all the loadables at once, then it starts running tests.
If the test gets a confidence of less than a certain threshold, it will pass the test onto the next UMD, which has the next largest partition. This is repeated until the threshold is reached or there are no more UMDs, at which point the output happens.

The key differences can be summed up as follows:

- Single-UMD: Repeats steps 3-7 for each partition required
- Multi-UMD: Repeats steps 3-5 for every partition, then repeats steps 6-7 for each test needed 

Anything else can be reasonably easily inferred from code, and any comments I may have left in the code (but there's not a lot as it's remarkably self-explanatory).

#### Custom Runtimes - Compilation
To compile the runtime, you have two options.
1. Set up an environment that will be able to perfectly compile the code with all outdated libraries being installed along with their dependencies and building the environment from the ground up (long process - not recommended)
2. Use another Docker

For the docker, I used the Ubuntu 16.04 docker. Then I needed to install the toolchain for ARM compilation.
````
docker run -it -v /home:/home ubuntu:16.04
apt-get update
apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu make -y
````
After the successful install, you can compile the runtime. First navigate to the following directory:
````
<your_path>/NVDLA_SlimNNs/Runtime/sw/umd
````
Within here is a ````.sh```` file: ````compile_and_transfer.sh````. will compile the runtime and output the file into ````Runtime/runtime_outputs````. 

Usage: ````./compile_and_transfer.sh [str: filename = nvdla_runtime]````

N.B. If you attempt to compile and transfer with the same name as one already in the destination directory, the original will be overwritten.

#### Custom Runtimes - Execution
Two runtimes (single and multi) are provided within the Runtime directory. These can be used and have their own help message in case of misuse. Included in the repository are also some scaled up (64x64) CIFAR10 images, for testing purposes.

To execute a runtime, you should first copy the runtime into the VP docker, which can be done with the ````docker cp```` command above. After copying, you can run it just like the original NVDLA runtime that comes within the onnc/vp docker. Recommended execution is as follows:
````
./runtime --parts <int: parts> --loadable loadable1.nvdla \
--loadable loadable2.nvdla --mean 0.4914,0.4822,0.4465 \
--normalize 0.2470,0.2435,0.2616 --image image.jpg --rawdump
````
The arguments can be broken down as follows:

- ````--parts````: the number of partitions the network consists of
- ````--loadable````: the loadable files for each partition
- ````--mean````: the mean values for each channel of the image - used for normalisation
- ````--normalize````: the normalisation values for each channel of the image - used for normalisation
- ````--image````: the image to be used for inference
- ````--rawdump````: the output file will be human-readable

If the image isn't the correct size you will get some strange errors. If the image is too small, it isn't catastrophic, but if it is too large, the runtime may crash (as C is so memory unsafe).

## Testing and Verification
### Functional Validation
Functional validation was shown through a selection of 100 images, each from CIFAR10, scaled up to be the correct format for the AlexSNN network. These images were run through the network, and then through both the ONNX and PTH versions of the network too. The results were that the NVDLA network had an error rate in the degree of 10^-3, as compared to the other forms. As this is across hardware, and the floating point memory size of the NVDLA is 16-bit, this is an acceptable error. Raw data can be seen in my dissertation file. 
### Performance
Performance testing was difficult, and there is likely some inaccuracy due to the measurement methods. [Insert diss stuff]