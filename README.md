# Chista
A web application for classification of images posted by social media influencers using deep learning algorithms

---

### About the Project

The ML modeling is implemented in three phases:

1. Classification of images within "Beauty" category including 6 sub-categories: {'eye', 'face', 'hair', 'lips', 'nail', 'products'}
* Using CNN-based binary classier for each sub-category
* Using an ensemble network to aggregate the results of the binary classifiers
2. Classification of 8 different categories including: {'beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel'}
3. Semantic segmentation of the images using a deep convolutional encoder-decoder architecture: [SegNet](https://arxiv.org/abs/1511.00561)

---

### Deployment Method

* A docker container is used to deploy the project on AWS.
* GitHub Actions are set up to push a container to AWS ECR.
* The container will be automatically deployed on AWS AppRunner after it is pushed into ECR.
* Flask is used to implement the backend code.
* The frontend is built using a simple HTML+CSS code to upload an image for classification.
* Tensorflow is used to create Deep Learning models (classifiers)

---

### To Run on Your Local Machine

* There are two ways to run the code on your local machine:
  1) Run Flask in your terminal:
     ```
     > flask run -h localhost -p <port_number>
     ```
  2) Run docker on your local machine:
     ```
     > docker image build -t <docker_image_name> .
     > docker run -p <port_number>:5000 -td <docker_image_name> 
     ```
---

### To Run When Hosted on AWS

* Simply use the the Default Domain provided by AppRunner on AWS.
