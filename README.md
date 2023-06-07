# Numerical_Method_Project

## Summary
In this research project, image processing and machine learning techniques will be applied to generate instructional images. The chosen method is the K-means algorithm, which is an unsupervised machine learning method used for cluster analysis. The goal of this algorithm is to divide a set of data points into k different clusters, such that each data point belongs to the cluster that is closest to it. By applying the K-means algorithm for color quantization, a color quantized image can be obtained. Additionally, the elbow method curve will be provided as a reference for the users. Afterwards, the Canny edge detection algorithm will be applied to the color quantized image. The Canny edge detection algorithm detects edges in an image through steps such as gradient calculation, non-maximum suppression, double thresholding, and edge tracking. It is effective in detecting fine edges while reducing noise and the impact of non-edge pixels.

## Steps
1. Import the original image in the initial interface.

![image](https://github.com/stupidfurfur/Numerical_Method_Project/assets/95353781/952702db-c23f-47bd-9b24-79b1ed4027c9)

2. Trigger the second button to obtain the elbow method curve for the image. The curve will automatically mark the inflection point, which represents the recommended optimal number of clusters.

![image](https://github.com/stupidfurfur/Numerical_Method_Project/assets/95353781/9e21a09e-b2d8-4a7e-8767-20b4957d5a53)

3. The third button is for color quantization. You can perform quantization according to the recommended optimal number of clusters or adjust the number of clusters manually.

![image](https://github.com/stupidfurfur/Numerical_Method_Project/assets/95353781/e797aa2d-cd8f-4cd6-8060-1e51cb0386fe)

4. Before implementing the drawing, save and name the color quantized image. This saved image will be used in the next step.

5. Import the specified color quantized image in the initial interface and adjust the parameters by dragging the trackbar to create a drawing practice image with less noise and prominent feature edges.

![image](https://github.com/stupidfurfur/Numerical_Method_Project/assets/95353781/993d0c20-0e82-4ead-ba3a-7d810164d28d)
