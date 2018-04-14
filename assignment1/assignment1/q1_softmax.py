import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    """

    ### YOUR CODE HERE
    x = x.astype(np.float128)
    if len(x.shape) == 0:
    	x = x.reshape(-1)
    if len(x.shape) == 1:
   	 x = x.reshape(-1, x.shape[0])
    x = np.exp(x)
    row_sums = x.sum(axis=1)
    #print x.shape
    #print row_sums.shape
    x /= row_sums.reshape(row_sums.shape[0],-1)
    ### END YOUR CODE
    
    return x

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002],[1001,1002]]))
    print test3
    assert np.amax(np.fabs(test3 - np.array(
        [[0.73105858, 0.26894142], [0.26894142, 0.73105858]]))) <= 1e-6

    print "You should verify these results!\n"

def test_softmax():
    """ 
    Use this space to test your softmax implementation by running:
        python q1_softmax.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    print "Passing a single scalar (1-D row vector)? Not sure why I'll ever get this - answer is 1"
    test1 = softmax(np.array(4))
    print test1
    assert np.amax(np.fabs(test1 - np.array(1))) <= 1e-6
    test2 = softmax(np.array([-2.3947641,0.89863696,-1.1773415,-1.0676703,-0.98260207,-1.0933431,-0.48529931,-0.70018922,0.71057025,1.3622189]))
    # 10.9607864016
    print test2
    assert np.amax(np.fabs(test2 - np.array([0.00832004071,0.22409458295,0.028108997,0.031367142,0.03415227,0.03057211,0.056155911,0.045297056,0.18567568,0.3562562]))) <= 1e-6
    print "All personal tests passed"
    ### END YOUR CODE  

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
