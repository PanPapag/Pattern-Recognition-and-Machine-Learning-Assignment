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
    
## Question 2 - Normalized Non-Negative Matrix Factorization
