# Positive/Negative Image Classifier
This is a machine learning project to classify images as either positive or negative.

## Overview 
This project trains a classifier to determine if a given image contains positive or negative phenomena.

The classifier is trained on a labeled dataset containing image samples and their lables (positive or negative). A deep neural network (DNN) model with an Convolutional Neural Netowrk architecture is used for the classifier.

## Dependencies
- Python 3.6 or higher
- Pytorch 1.10 and up.
- scikit-learn
- NumPy
- Pandas
- Anything else in requirements.txt is not mandatory but recommended.

## Data 
You can use whatever binary classification image-based dataset you have.


## Usage

To train the model:

```
python main.py 
```

Please edit the config.py file for paths and hyperparameters before you run it.

To classify a new image sample: 

```
Will be soon pushed to the repository.
```

Replace the image sample with your own image to classify. 

The output will be the predicted sentiment: `Positive` or `Negative`.

## Performance

Depends on your dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

## Acknowledgments

- This project was written by Noam Bergman
- The CNN model code was adapted from timm library
