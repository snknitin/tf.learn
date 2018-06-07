# tf.learn
Learning to use TensorFlow for Deep Learning



## Installation of the Environment

Get the yml file from [here](https://www.dropbox.com/s/k4i3gmo0bvss7g7/linux_tfdl_env.yml?dl=0)

    conda env create -f linux_tfdl_env.yml
    source activate tfdeeplearning
    source deactivate tfdeeplearning
    
    
## Crash Course

* Numpy
    * Arange is like range , start and stop with a step size exclusing the stop
    * linspace creates a linearly spaced array with start, stop and number of elements in the array
    * random seed needs to be executed each time , otherwise it might lead to different random values
    * argmax and argmin will return the index of the array where the max and min elements reside
    * When using matrices, you can use a mask or filter value as mat > 5 which turns everything into Boolean
    * To get the actual values mat[mat>50] will give the values from this filter
