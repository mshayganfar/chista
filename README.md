# Chista
A web application for semantic understanding of social media image posts using deep learning algorithms

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

* A docker container is used to deploy the project on the cloud (e.g. AWS)
* Web services are implemented using FastAPI framework
* Swagger is used for API development process in Python
* Tensorflow id used to create ML/DL models
