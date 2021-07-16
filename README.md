  # Recognizing Defect Patterns in Wafer Maps
  
  <p align="center">
    Inspection equipment for the semiconductor industry saves companies millions of dollars.
    This project uses the <a href="http://mirlab.org/dataset/public/"><strong>MIR-WM811K Corpus»</strong></a> to build a CNN classifier to automate classification of wafer defect patterns.
  </p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About the Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
      <ul>
        <li><a href="#repository-structure">Repository Structure</a></li>
      </ul>
    </li>
    <li>
      <a href="#data">Data</a>
    </li>
    <li>
      <a href="#project-details">Project Details</a>
      <ul>
        <li><a href="#data-wrangling">Data Wrangling</a></li>
      </ul>
      <ul>
        <li><a href="#eda">EDA</a></li>
      </ul>
    </li>
    <li>
      <a href="#modeling">Modeling</a>
    </li>
    <li>
      <a href="#results">Results</a>
    </li>
      <ul>
        <li><a href="#smote">SMOTE Oversampling</a></li>
      </ul>
      <ul>
        <li><a href="#final-results">Final Results</a></li>
      </ul>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About the Project


### Built With
<b>Python 3.9</b>:

Data modules
* pandas
* numpy

Plotting modules
* matplotlib
* seaborn
* scipy.stats

Misc. modules
* math
* warnings
* datetime
* itertools

Custom functions found [here](./src/).

### Repository structure
* img: Figure image files
* notebooks: Jupyter Notebooks
* src: Python files

A presentation summarizing the data analysis and results can be found [here](insert pdf link).

## Data
This project explores over 800,000 real wafer maps from the MIR-WM811K Corpus, made public by the MIR Lab at the National Taiwan University:
* [MIR-WM811K Corpus](http://mirlab.org/dataset/public/MIR-WM811K.zip)
  * [Pickle Version](https://www.kaggle.com/qingyi/wm811k-wafer-map)

## Project Details

### Data Wrangling
The data started in the form of a pickle file, so I converted it to a pandas DataFrame for easier processing. The dimensions of the DataFrame were 811,457 rows by 6 columns. The first column contained the main data, which had entries in the format of 2-D numpy arrays of the wafer maps, with each element labeled as 0, 1, or 2, for failed, functional, and space outside of the wafer. I combined the functional and empty space categories to make the maps binary. Most of the samples were unlabeled, and among the labeled samples, only 25,519 were classified as "failed." I dropped all but these 25,519 rows, under the assumption that each wafer can be flagged ahead of time as failed by applying a simple chip yield threshold.

![image example](./img/hospitals_time.png)

Next, I dropped the extraneous columns of dieSize, lotName, and waferIndex. The dieSize information is already contained within the wafer map dimensions. The failureType labels were in the form of strings in nested lists, so I converted them using a categorical encoder. I also converted the wafer maps into a form suitable for Keras and tensorflow. The dataset came with training/test set labels, so I separated the data by these.

### EDA
First, I explored the wafer maps, by plotting them as images. Below are examples from each failure type:
![image example](./img/hospitals_time.png)
The failure types have patterns, such as lines for the "scratch" class, and outer rings of failed chips for the "edge-ring" class. I noticed that the dimensions of the maps were not uniform, with variation across and within classes. Dimensions varied in the range of roughly 10x10 to more than 100x100. I resized all of the images to a uniform size, and made the size an adjustable parameter.

Next, I explored the class distributions, and was suprised to find significant class imbalances. 
![image example](./img/hospitals_time.png) ![image example](./img/hospitals_time.png)
![image example](./img/hospitals_time.png)
For instance, "edge-ring" accounted for nearly 50% of the data, and "near-full" (nearly all failed chips) was 0.3% of the data. Additionally, the training and test datasets had different distributions. This adds challenge to the problem, but also makes it more realistic, since the real-world will not follow the training set distribution. I decided to focus on optimizing the metric of recall, since high accuracy and precision can be obtained by ignoring the underrepresented classes

## Modeling 
I trained a convolutional neural network (CNN) for this classification project, because it is typical in the application of image recognition. I adapted a simple CNN model used for recognition of the MNIST handwritten digits corpus. A model diagram of this CNN model can be found [here](./img/). An important feature of the model is the final activation function is a softmax, which is useful for multiclass classification problems because it assigns a probability of each sample belonging to each class

## Results
I trained the CNN model by testing ranges of (hyper)parameters including optimizer type, learning rate, number of epochs, batch size, kernel size, and image size. The procedure I followed was to train and evaluate the model with k-fold cross-validation (k=5), which was useful for hyperparameter tuning. Then, I trained the model on the entire training dataset, and evaluated its performance on the test set. I included the metrics of recall, precision and accuracy to be informative, but I only focused on optimizing the recall.

### SMOTE Oversampling
Without tuning the parameters, I found that the initial results of testing could use improvement. In particular, the model performed significantly worse on the test set than the validation set. I attempted to address this with SMOTE oversampling, to oversample the underrepresented classes in the training set. This generated synthetic samples using a variation of a k-Nearest Neighbors (k-NN) algorithm. I oversampled until all classes had as many samples as the plurality class.

However, once I tested the modeled by training with the SMOTE oversampled data, it resulted in significantly worse recall and accuracy, and marginally improved precision. A comparison is below:

Based on this, I scrapped the idea. However, the class imbalance issue could be further explored in the future using techniques such as imbalanced class weights or augmented training data generators.

### Final Results
After testing various hyperparameters I settled on the values below, which resulted in the best performance using the metric of recall.

  ![image example](./img/hospitals_time.png)

Note that the validation set performs better than the training set, which is not an error, but is likely due to the sensitivity of sampling of the underrepresented classes for training and validation.
Here were the final results:

While this value can be improved by training longer, using more data, etc., it is an impressive result given the severe class imbalance. CNN models like this have many applications in the inspection equipment subset of the semiconductor manufacturing industry, and companies like KLA-Tencor and Applied Materials have had a recent push to apply data science toward automating quality control processes.

<!-- Contact -->
## Contact

Author: Christopher Shaffer

[Email](christophermshaffer@gmail.com)

[Github](https://github.com/chrisshaffer)
