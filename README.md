# Pattern-Recognition-and-Machine-Learning-Assignment

## Question 1 - Preprocessing, dimensionality reduction, visualization, and classification of images

**Data**: The data set consists of 30 RGB color images that depict landscapes during the spring, fall, and winter (10 images for each season). The first letter in the name of each image file determines the season in which the image was recorded, for example, image F1.jpg was recorded in the fall, while image W10.jpg was recorded in the winter. Therefore, the naming of the files fully determines the category to which each image belongs. The images consist of different numbers of pixels. Each pixel consists of three color values ranging from 0 to 255, which determine the brightness intensity of red, green, and blue at each point in the image. The data is available in the file images.zip.

Tasks:

1. Write a function loadImages(path) that takes as input the path to the folder containing the images (e.g. loadImages("C:/images")), reads the images, converts them to 100 x 100 pixels, and returns a data array of 30 columns, where each image is represented as a column vector. The function should also return the labels (coded as integers) for the different categories to which the images belong (e.g. 0 for images recorded in the winter, 1 for images recorded in the fall, and 2 for images recorded in the spring).
2. Write a function PCA_ImageSpaceVisualization(X) which takes as input a data array, calculates the first two principal components of the data and plots the data in the first two principal components. The function returns a plot in which the images are visualized in the two-dimensional space resulting from the projection of the data onto the first two principal components.

    2.1 What does it mean when images are close to each other in this two-dimensional space depicted in the above plot? What does it mean when images are far from each other? Can we generalize these conclusions for the original high-dimensional space of the images?

    2.2 Do the images corresponding to one of the seasons tend to cluster together more closely than the others? Why does this happen?

3. Compare the accuracy of the 1-NN nearest neighbor classifier and the linear support vector machine (SVM) in the problem of recognizing the season in which an image was recorded. In other words, compare the performance (in terms of classification accuracy) of the above classifiers in the classification of image data into the categories of winter, spring, and fall. Address the classification problem using 1) the original high-dimensional images in vector form, and 2) low-dimensional features extracted through PCA.

    3.1 Define mathematically the measure of classification accuracy.
    
    3.2 Use 5-fold cross validation and report the average classification accuracy for both classifiers for both the high-dimensional data and the low-dimensional features.
    
    3.3 How will you determine the dimensionality of the features that will be extracted through PCA?
    
    3.4 Which classifier has the best performance and why?
    
## Question 2 - Regularized Non-Negative Matrix Factorization
Consider the following optimization problem for regularized non-negative matrix factorization (regNMF):

$$\min_{W,C} \lVert X - WC \rVert_F^2 + \lambda \lVert W \rVert_F^2 + \lambda \lVert C \rVert_F^2, s.t. W \geq 0, C  \geq 0$$

The problem clearly has no closed-form solution, and therefore must be solved iteratively. Implement in the function RegNMF(X,k,lambda,epsilon) an iterative algorithm for solving the above optimization problem. The function takes as input a non-negative matrix X of dimensions d x N, the number of components k, and the value of the regularization parameter lambda and the termination threshold epsilon, and returns the non-negative matrices W of dimensions d x k and C of dimensions k x N.

To determine if an iterative algorithm is converging to the optimal solution, we usually follow the reconstruction error ||X - W[t]C[t]||_F^2/||X||_F^2 at each iteration, and if the change between two consecutive iterations is smaller than a threshold e (||X - W[t]C[t]||_F^2 - ||X - W[t-1]C[t-1]||_F^2 )/||X||_F^2 < e, with e being 0.01 or 0.001 or 0.0001, we terminate the algorithm. t in the above relationships represents the iteration index.

You are asked to study the convergence of the algorithm using synthetic data. Specifically, construct a random matrix X of dimensions 500 x 1000 with non-negative values, and study the behavior of the regNMF algorithm with respect to the number of iterations required for convergence, for k = 1, 10, 100, and e = 0.1, 0.01, 0.001. What are your conclusions about the behavior of the algorithm for different values of k and epsilon (e)?
