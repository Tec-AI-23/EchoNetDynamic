### [↴ Jump to theoretical background](#left-ventricle-automatic-segmentation-with-deep-learning-on-the-echonet-dynamic-dataset)

## Pre-requisites

Make sure to have a Python version between >=3.7,<3.11. **PyTorch does not work on 3.1!**

We recommend 3.11.

## Dependencies installation

Initialize a Python virtual environment (venv) according to your respective os. Make sure the venv folder is called `.venv`.

### Windows:

#### Create venv
```sh
python -m venv .\.venv
```

If multiple Python versions are installed, it is helpful to use:
```sh
py -3.11 -m venv .\.venv
```

Change `-3.11` with the desired version.

#### Start venv
```sh
.venv\Scripts\activate
```

### Linux:

#### Create venv
```sh
python3.11 -m venv .venv
```
Change `python3.11` with the desired version.

#### Start venv
```sh
source .venv/bin/activate
```
## Install dependencies

Run:
```sh
pip install -r requirements.txt
```

### Add new dependencies 

To further add new dependencies first write down the name of the dependency on the file `PACKAGES_NAMES.txt`. Make sure to leave a blank line at the end of it.

Then, run the script `requirements_generator.py`, which fetches the latest stable versions of each dependency from the `pip` website and then with each fetched version it writes the `requirements.txt` file.

However, if a certain version that is not the latest is needed, **DO NOT RUN `requirements_generator.py`**! This is just a temporal automation.

# Left Ventricle Automatic Segmentation with Deep Learning on the EchoNet Dynamic Dataset

#### Team Members

- Omar Alejandro Rodríguez Valencia
- Cristian Javier Cázares Molina
- Jair Josué Jimarez García
- Musel Emmanuel Tabares Pardo
- Siddhartha López Valenzuela

## Introduction
Using the EchoNet Dynamic dataset, that contains a lot of images of different patient's echocardiograms, with Image Segmentation and Deep Learning (DL) techniques, we want to be able to predict the masks of the area of the left ventricle of every echocardiogram image. This can be helpful for medical experts to extract information and diagnose patients with greater ease.

### Scope
We want to determine the better model for this situation, between a Landmarks Model or a Masks Model. We choose the best model based on the accuracy of the masks generated by the model, that will be evaluated with new images that contains more noise than the ones that have been used for training. The training dataset contains clinical measurements and other information that will be ignored.

### Background
In the past, DL techniques have already been used in the medical field, for example, to detect cancerous cells [^1]. Usually, this projects that requires image segmentation, use the "Fully Convolutional Neural Networks"[^2], that focus on assing a class to each pixel of the image, and not label the image as whole. Past investigations have done masks of the heart's left ventricle, but they used magnetic resonance images, and they got acceptable results (87.24% dice score), so this suggests that our problem can be solved with a Fully Convolutional Neural Network.

## Data
The Stanford University data set have 10,030 echocardiography videos (approximately 7.9 GB) since 2016 until 2018 as part of routine clinical care. Each video has 100 frames, approximately; and every video was cropped to remove the text and information that is not relevant for the scanning process. All frames were resized into 112X112 pixels, and some frames were matched with their respective masks, that will be used to train the model. The masks are given as coordinates that represent essential points to identify the left ventricle at the echocardiogram image.

![image](https://github.com/Tec-AI-23/EchoNetDynamic/assets/83721976/bf563538-a787-429a-9169-a594d3a0e808)

Like we said before, the data set contains clinical measurements that will be ignored, since we only want to generate masks tracing the left ventricle. As the data is from the Stanford University, we assume that it follows laws concerning privacy on medical data. The data set can be found in the Stanford web page. The data set will be stored only locally to avoid any possible disturb.


## Methodology

### Deep Learning
This methodology makes use of a perceptron, which is given any number of input characteristics and balances the weights of the inputs to get as close as possible to the expected output. The main difference between a Deep Neural Network (DNN) and a simple perceptron model is that the DNN layers contain any number of neurons, which allows each layer to extract features and interpret the data in different ways.

### Convolutional Neural Networks
Better known as CNN, is a Deep Learning technique that process images. An important factor about CNN is its convolutional layers, which use filters to extract features from the image.  They also consist of pooling layers that are responsible for reducing the dimensionality of the image, to process it more easily and efficiently.  Within this category, there is another type of layer, the max pool layer, which takes the maximum value of a pixel area of the image to create a new smaller image.

![image](https://github.com/Tec-AI-23/EchoNetDynamic/assets/83721976/6700636f-46ce-442f-b135-1b602eb8a58f)

### Software and Hardware
To create the model, we used PyTorch, which is a framework that specializes at designing and implementing deep learning models in Python. The PyTorch version is 2.1.0. The computer where the model was run and tested was a laptop with 8 GB RAM
and a 3070 Ti NVIDIA graphic card.

### Architecture Used (U-Net)
We used a U-Net[^3] architecture, that was created to solve biomedical image processing tasks, so its designed to solve problems similar to ours. The U-Net doesn't require a lot of data to be trained and it has some characteristics that capture both local and global image features.

The next figure shows a fully convolutional network, it assigns a class to each pixel in an input image, and consists of two paths: contracting and extensive. The contracting path it is a sequence of 3X3 convolutions, the application of ReLu functions and 2X2 max pooling with stride 2 for downsampling. The extensive path then upsamples the feature map to revert the changes that were made in the contracting path in order to increase the resolution of the output.

![image](https://github.com/Tec-AI-23/EchoNetDynamic/assets/83721976/4148ac70-0382-44ec-92a8-cdaabd31c16d)

### Mask Approach
The model will receive the echocardigram image and will return the mask of the left ventricle. The mask is an image that can only take 0's and 1's as values for each pixel. To train the model we will need to create the mask with help of the above mentioned coordinates. Since the predicted masks need to be compared with the actual masks, we will use the Dice Score formula. Which compares every pixel of both images and return a number value between 0 and 1, the closer to 1, the better the mask.

$$Dice = \frac{2 * |X \cap Y|}{|X|+|Y|}$$

### Landmark approach
In this approach, the model will be fed with the original image, and will return a tensor as a probability map of which pixels could be a landmark. Landmarks are key points in the shape of the image we want to identify. The way to identify these landmarks depends on the problem, but the goal is that a set of landmarks can cover as much as possible the area to be identified.

Given the situation that there are objectively no correct landmarks, even if made by a human, this approach becomes complicated. Because by training the model with specific coordinates, the model will consider those to be the only correct landmarks and will correct for others that might be considered different but correct. To avoid this situation, we will give the model a probability map during training to correctly calculate the backpropagation, and thus only correct landmarks that are far away from the edges or key points of the figure.

In order to create a mask, we need at least 3 landmarks. The problem is that the veracity of a 3-point mask would only be a triangle and would not correctly cover the area of the figure, even if the landmarks are correct. On the other hand, increasing the number of landmarks too much, also increases the processing weights of the model algorithm to be trained, increasing the time significantly. So we opted for 7 landmarks because it gives us enough information about the curvatures of the figure and in this way we do not exceed the weights of the model, but this doesn't mean that this is really the optimal number.

![image](https://github.com/Tec-AI-23/EchoNetDynamic/assets/83721976/5873eb09-fd0e-4cd2-a184-dbc60a58be37)
![image](https://github.com/Tec-AI-23/EchoNetDynamic/assets/83721976/7622f0c8-1e5d-413d-892b-a6110184c8c1)

## Experiments
Below we will show you in detail some of the problems we faced during the development of this project.

### Mask extraction methods
As explained above, we had to create the masks ourselves from a set of coordinates. To address this, we experimented with 3 methods to make the masks, but all of them were implemented with the OpenCV function, "fillPoly". This function requires as parameters, the image where the polygon will be drawn, the points of the polygon (in this case, our coordinates), and the color.

The problem is that the coordinates were not sorted, so if we entered the set of coordinates as we received it, we obtained different types of figures that did not work as masks. So we had to experiment with how we sorted the points. The following are the methods we used.

- **Simple Sort Method:** Using "sorted" Python function, which takes an iterable and returns it, but sorted in ascending or descending order.

- **Centroid Method:** Calculating the mean between all the points, we created a centroid. Using that centroid and the "arctan2" Numpy function, we created a sorted array.

- **Convex Hull Method:** Using the OpenCV function, "convexHull", which scans the array of coordinates using the Sklansky algorithm and sorts them.

### Heat maps methods
The landmark approximation required us to formulate the best possible heat map, so we had to try different methods.

- **Pixel expansion:** In order to make the network predict the landmark close to the given one, the acceptable area should be narrowed down. To make that possible, the pixels given by the dataset are "expanded" so that their size is greater than only the one pixel they initially represent.

- **Border limits:** We needed the model to be able to differentiate between a point close to the given landmark but outside the ventricle border and one inside the ventricle border. So we had to delimit the edges so that it would not predict outside that area. To do this, we use two morphological operations from the OpenCV library: erosion and dilation. Erosion is a filter that given a pixel, analyzes the value of n-number of pixels around it, if the value of all those pixels around it is 1, then the given central pixel will also be 1, otherwise, it will be 0. Dilation is a filter that works similar to erosion, but it will only give 0 to the given central pixel if the pixels around it are also 0, otherwise, it will be 1. So what these two operations help us with is that dilation makes the area of the mask grow, and erosion makes the area shrink, and with an XOR operation we compare them and get a final image that gives us the outline.

- **Euclidean distance:** We calculate the Euclidean distance between the given landmarks and the rest of the pixels in the image. Each landmark gives us a map that we multiply to the previous map given by pixel expansion and border limits. In this way, the resulting map only decreases gradually in the areas near the given landmarks and the edge of the figure.

$$Euc = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$

- **Gaussian distribution:** With the "gaussian" function of the Scikit-Image library and apply a Gaussian filter, which given a specific area, marks the central pixel as the highest value and as it moves away from the center, it takes lower and lower values, creating a Gaussian distribution. Once given the map, we multiply again with the previous maps and finally define correctly the area in which the model can predict the landmarks.

### Optimal landmarks
As mentioned earlier, the number of landmarks is something we have to consider because of the computational processing required. With this in mind, we wanted to determine the most important landmarks to keep the mask quality, no matter the number of landmarks. The method we used was to calculate the angle between one landmark, the past one, and the next one. The landmarks with smaller angles are the most important ones, and as the angle is closer to 180° it can be ignored without losing important information. 

[^1]: Nasim, M. A. A., Munem, A. A., Islam, M., Palash, M. A. H., Haque, M. M. A., & Shah, F. M. (2023). Brain tumor segmentation using enhanced u-net model with empirical analysis.
[^2]: Long, J., Shelhamer, E., & Darrell, T. (2015, June). Fully convolutional net-works for semantic segmentation. In Proceedings of the ieee conference on computer vision and pattern recognition (cvpr).
[^3]: Ouyang, D. (2020). Video-based ai for beat-to-beat assessment of cardiac function. Nature.
