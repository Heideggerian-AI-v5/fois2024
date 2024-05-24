# fois2024
Code associated to a FOIS 2024 conference submission

**Getting started**

***Installing dependencies***

Git clone this repository then open a command shell and change to the repository clone directory. Once there, run

`
    pip install -r ./requirements.txt  
`

to install the dependencies. Most of these should be straightforward to install via pip. Pybullet may be an exception on Windows, as it requires also the compilation of C/C++ code and, depending on how your computer is set up, this may not be immediately available. 

For more detailed instructions on how to install pybullet on Windows, see:

[https://deepakjogi.medium.com/how-to-install-pybullet-physics-simulation-in-windows-e1f16baa26f6](https://deepakjogi.medium.com/how-to-install-pybullet-physics-simulation-in-windows-e1f16baa26f6)

***Building shared library for contact mask calculation***

Once all dependencies are installed, you can try building our contact.cpp file into a shared library. Assuming the cmake command is available on your system, it is enough to change to the turtle_sim folder of the repository and perform the following sequence of commands:

`
    mkdir build  
    
    cd build  
    
    cmake ..  
    
    cmake --build .  
    
    cp libcontact.* ..  
`

Note that we included two built binaries: libcontact.so (for Unix), libcontact.dll (for Windows) so you can also try to skip the make step.

**Part I: Collecting training data for object detection**

(Note: we include a pretrained model for functional parts in the repository, so you can also skip to part 3.)

In this part of the experiment, you will drive a turtlebot through a simulated world. For now, the robot cannot control its own motion -- this is where you will step in -- however it does control its own perception. It reasons about what to keep track of, and whether to store something in longer term memory, on its own based on an overarching goal it has. For this experiment, the goal is to keep track of support situations.

You will see the robot from a 3rd person perspective, as well as see what the robot's camera sees. The robot's camera image is also annotated with features for optical flow (magenta points, where the flow is represented by magenta lines) and contact masks (yellow). You will also see an image of the recognized object masks, where each object is represented by a different color.

As you drive, the robot will "pay attention" to the incoming visual data, looking for situations of support. If it does see such a situation, it will store a frame of what its camera sees. We recommend that where you see a support situation, to try to images from several angles and distances.

To run this part of the experiment, you will need two command shells. In the first one, go to the repository clone directory and run

`
    python runTurtle.py -g -l supportScene.json  
`

This will start up a simulated world with a turtlebot and some objects. You can drive the turtlebot using the arrow keys.

Then in the second command shell go to the turtle_sim folder of the cloned repository and run:

`
    python partI_collectData.py  
`

This will store the frames at the location ../data/raw/ but if you want the frames stored somewhere else, you can instead run

`
    python partI_collectData.py -dest <path/to/your/storage/location>  
`

The robot will only store "relevant" frames, i.e. frames where it sees something it believes is an instance of a support relation.

**Part 2: Training a YOLO object detection model**

(Note: we include a pretrained model for functional parts in the repository, so you can also skip to part 3.)

If you have an instance of runTurtle.py running, you should close it now -- it is not necessary for this step, and closing it frees some computational resources. This part will need all the computer power you have available, and expect it to take several hours.

In a command shell, go to the turtle_sim folder of the repository clone and run

`
    python partII_trainYOLO.py  
`

This will assume default locations for training data (../data/raw/) and location for the resulting model (../model/supp/model.pt). If you want instead to use your own, run:

`
    python partII_trainYOLO.py -i <path/to/your/image/storage/location> -m <path/to/your/model>  
`

**Part III: Recognizing functional parts**

You will need two command shells, like in part I. In the first shell, go to the repository clone directory and run:

`
    python runTurtle.py -g -l supportTestScene.json  
`

This is a similar scene to the first, but the objects are not hanging on hooks anymore.

In the second command shell, run:

`
    python partIII_findSupportParts.py  
`

This will use an object detection model placed at the default location ../model/supp/model.pt but if you want to specify another location for the model, run:

`
    python partIII_findSupportParts.py -m <path/to/your/model>  
`

As before, you will have to drive the robot but it will control its own perception, this time based on the goal to look for functional parts of objects -- specifically, those involved in supporting (via hooks).

There are no frames being saved this time, but you will see, apart from a 3rd person perspective on the robot, also an image of what functional parts the robot recognizes in the current image.
