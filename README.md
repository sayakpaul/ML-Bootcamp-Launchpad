
# Machine Learning Bootcamp

Contains notebooks prepared for a Machine Learning Bootcamp organized by [Google Developers Launchpad](https://developers.google.com/community/launchpad).

Accompanying deck link: http://bit.ly/mlb-sayak. 

## Acknowledgements:
- The entire team at PyImageSearch
- [Martin Görner](https://twitter.com/martin_gorner)
- [Yufeng Guo](https://twitter.com/yufengg)
- ML-GDE team for granting me GCP Credits to aid this Bootcamp

## Motivation
Typical machine learning steps:

  

-   Problem understanding
    
-   Data collection and understanding
    
-   Data preprocessing
    
-   Model building and training
    
-   Hyperparameter tuning, revaluate model and repeat
    
-   Model deployment
    
-   Serving prediction and model monitoring
    

  

Optionally, you might want to optimize your model with respect to its target deployment environment and then deploy it. In this Bootcamp, we are going to take a look at each of the above steps. We will be using TensorFlow (2.0) as our main library for doing machine learning - we will be doing deep learning which is another form of machine learning only. Specifically, we will be using TensorFlow’s high-level Keras API for model building and training.

  

We will be using the [Flowers-17](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/) dataset. So, for a given flower an ML model needs to predict the most likely category of the flower. In machine learning and computer vision literature, this problem is commonly referred to as an image classification problem and it kind of dominates the kind of applications that have been made possible by the intersection of machine learning and computer vision.

  

We will also be using several Google Cloud Platform (GCP) services like AI Platform, Google Cloud Storage. But note that all of the typical machine learning steps we saw earlier can be performed on your local computer. So why use cloud services and incur that additional cost? That is because your enterprise does not operate on a local computer. It is most likely to serve tens of thousands of end-users. Your local machine might not have that bandwidth to support this kind of scaling whereas cloud providers like GCP offer you off-the-shelf and specific things to achieve whatever scale you want to cater to. Typical cloud functionalities that come to aid you in this regard are -

  

-   You might want to autoscale your infrastructure to be able to speed up the inference time of your ML model.
    
-   You would really want to add a strong authentication layer just before your model is consumed as a REST API.
    
-   You would want to orchestrate the version controlling of your models.
    
-   You don’t have the on-premise infrastructure to support large training workloads of your model. So, you might want to go for Cloud-based VMs that support GPUs easily.

And [much more](https://cloud.google.com/blog/products/ai-machine-learning/how-to-serve-deep-learning-models-using-tensorflow-2-0-with-cloud-functions).

AI Platform helps you achieve all of the above in a pretty customizable way as we will see in this Bootcamp. It does not encourage just one click deployments but it provides you with enough tools to tweak the novel bits necessary for your deployment and prototyping purposes. 

## Agenda:

- **Problem**: Given an image of a flower, the system would predict the category to which the flower is most likely gonna belong to. 

- **Data available to represent the problem space?** - Yes, **Flowers-17** dataset. 

- **Data collection**
	- Collect the Flowers-17 dataset
	- Visualize the images and the labels
	- Create train and test sets

- **Data preprocessing**
	- Scale the pixel values of the images 
	- Encode the labels

- **Data input pipeline**
	- Data augmentation with `ImageDataGenerator`
	- Measure the performance of `ImageDataGenerator`
	- Using `tf.data` to speed things up
	- Building a data input pipeline with `tf.data`

- **Model building and training**
	- Starting with a shallow convnet
	- Analyze model training and model performance
	- Doing better with transfer learning (ResNet50)
	- Analyze model training and model performance

- **Tuning the learning rate hyperparameter and faster training**
	- Using the cyclical learning rate to train  our model better
		- Learning rate finder
		- CLR callback
	- Using mixed-precision training to speed up the training process

- **Model deployment**
	- Serializing the final model in SavedModel format
	- Uploading our model artifacts to GCS
	- Kickstart an AI Platform model submission job via the AI Platform GUI

- **Serving predictions**
	- Randomly selecting a set of images for testing
	- Preparing the images for online prediction
	- Using AI Platform’s predict jobs to perform inference

We will be using TensorFlow 2.0, Google Cloud Platform and Weights and Biases and Python 3 (of course!). Let's get started! 

![](https://i.ibb.co/c6Rfn9j/Untitled-Diagram-1.png)
