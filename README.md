# CSC 7210: Project 2
## Katherine Brown

### About
This project casts the problem of the computational detection of diabetic retionopathy as a anomaly detection problem. I hypothesized that retinal scans with diabetic retinopathy would contain enough artifiacts to be considered anomalous. 

Unfortunately, this was not the case. I was able to develop the autoencoders to detect anomalies in other image datasets, but the autoencoders failed on the diabetic retinopathy data. The issue I believe is twofold: 
1. The diabetic retinopathy images consist of the same general shape. When debugging, I noticed that the autoencoders re-constructed images based on shape and color.
2. The clinical abnormalities used to diagnose diabetic retinopathy are minute. The error of missing these features is small and overshadowed by the remaining reconstruction.

### Datasets Used
1. [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)
2. [Fruit 360](https://www.kaggle.com/moltean/fruits). I trained the autoencoders on apples and attempted to reconstruct bananas

### Required Package Installation
You will need to install the following packages to run any file in the submission. Keras, PyTorch, and Scikit-Learn implement the anomaly detection models. Matplotlib produces the visualizations.

`conda install pandas=0.25.1`
`conda install numpy=1.17.2`
`conda install matplotlib`
`conda install scikit-learn`
`conda install tensorflow`
`conda install keras`
`conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch`
`conda install torchvision`