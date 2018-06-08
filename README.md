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
* Pandas
    * Dataframe has a method as_matrix() which converts it into a matrix to be used in numpy
    * They have inbuilt plot functionalities as well. df.plot(x,y,kind='scatter') will create a scatter plot between two columns x and y
    
* Matplotlib
    * To see the visualizations we use **%matplotlib inline** in Jupyter
    * xlim and ylim functions set the limits for the range of axes in plots
    * Visualize a matrix using plt.imshow(mat,cmap='') and use a particular colormap 
    * plt.colorbar() will give you a legend for the imshow colormap 
* Scikit-learn
    * MinMaxScalar can be used to fit and transform the data for normalization to [0,1] range
    * Train test split can be done easily
    * This package supoorts everything except neural networks


## OOP Concept

When you extend a class or inherit from a super class, to access the first class's init method we use super().__init__(args) to access the super class 
