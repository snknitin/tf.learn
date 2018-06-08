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
    * df.sample(n=250) to sample random rows from the dataframe
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

When you extend a class or inherit from a super class, to access the first class's init method we use super().\_\_init\_\_(args) to access the super class 

## TensorFlow

* Syntax or useful commands
    * tf.fill((dimensions),scalar value) or use tf.ones and tf.zeros without the scalar parameter
    * tf.random_normal((dimensions),mean,stddev)
    * tf.random_uniform((dimensions),minval,maxval)
    * Call sess =tf.InteractiveSession() to evaluate operations outside the session. Useful in Jupyter notebook
    * a.get_shape() will give the TensroShape with dimensions
    * tf.matmul(a,b) Matrix multiplication
    * tf.multiply(a,b) is element-wise multiplication
* Graphs
    * when you start TF, a default graph is created which can be accessed as a GraphObject using tf.get_default_graph()
    * g=tf.Graph() to create another graph
    * with g.as_default():  This makes it the default
 * Variables and Placeholders
    * These are the two types of Tensor objects
    * Placeholders are initially empty but they need to be declared with an expected data type(like tf.float32) and shape of the data
    * You get a error if you try to run a variable without initializing. Use init = tf.global_variables_initialzer() and sess.run(init)
 * Regression and Classificaton tasks
    * error = tf.reduce_sum() and tf.reduce_mean() to get cost functions
    * optimizer = tf.train.GradientDescentOptimizer(learning_rate) and train = optimizer.minimize(error)
 
 * Estimator Object
    * tf.estimator is an API with several models which resembles scikit-learn ML models
    * Feature columns -  tf.feature_column.numeric_column('x',shape=[1])
    * Use train_test_split to get x_train
    * Input function -  tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size,num_epochs,shuffle=True)
    * estimator.train(input_fn,steps=1000) This gives global step-wise loss
    * trained_metrics = estimator.evaluate(input_fn,steps)  This gives a loss, average_loss and global step





