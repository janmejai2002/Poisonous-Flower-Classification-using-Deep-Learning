# Poisonous Flower Classification using Deep Learning
A deep learning model to classify the image of a flower into one of the five categories : Calotropis, Cascabela, Datura, Hemlock, Nerium.
There are various types of poisonous plants which exist in nature and biologists or even common people can classify them using certain characteristics such as smell, stem color, the flowers they bloom etc.

If a person unknowingly comes in contact of a poisonous flower or mistakenly consumes it, their treatment becomes much easier if the type of flower was known.

The aim of this project is to use Deep Learning to classify the image of a flower of a poisonous plant into one of the following categories:
![](https://github.com/janmejai2002/Poisonous-Flower-Classification-using-Deep-Learning/blob/master/examples/collage2.jpg)

1.  [__Calotropis__](https://en.wikipedia.org/wiki/Calotropis_gigantea#:~:text=Calotropis%20is%20a%20poisonous%20plant.&text=It%20is%20used%20as%20an,keratoconjunctivitis%20and%20reversible%20vision%20loss.)

2.  [__Cascabela Thevetia__](https://www.childrens.health.qld.gov.au/poisonous-plant-yellow-oleander-cascabela-thevetia/#:~:text=Cascabela%20thevetia%20is%20a%20restricted,particularly%20the%20fruit%20and%20seeds.&text=Symptoms%20may%20include%20a%20burning,a%20slow%20or%20irregular%20heartbeat.)

3. [__Datura__](https://en.wikipedia.org/wiki/Datura#:~:text=All%20species%20of%20Datura%20are,even%20death%20if%20taken%20internally.)

4. [__Hemlock or Conium maculatum__](https://en.wikipedia.org/wiki/Conium_maculatum)

5. [__Nerium oleander__](https://en.wikipedia.org/wiki/Nerium#Toxicity)

While these flowers look really beautiful, they are certainly nasty if mistakenly consumed.

## Get it running on your PC

Go to the desired directory and open terminal there. For windows a quick way to open cmd in any folder is to type `cmd` in the navigation bar and pressing `Enter`.

Run the following command

        git clone https://github.com/janmejai2002/Poisonous-Flower-Classification-using-Deep-Learning.git

(For windows users this will only run if you have git utilities installed for cmd. If this does not work use the default shell git provides)

I am using `python=3.7` and the other dependencies are inside `requirements.txt` .

To setup your environment run the follwing code below after creating a new environment.

        pip3 install -r requirements.txt

#### The repo:

-  `dataset` contains images of the five classes I mentioned above

-  `examples` has some sample images.

-  `model` contains the python file of the model class. I have used transfer learning and used MobileNetV2.

- `results` has screenshots of sample predictions I made

- `bing_search_api.py` lets you collect your own dataset, I have explained it further below.

- `classify.py` is the script to make predictions.

- `train.py` is the scipt for training the model.

- `lb.pickle` contained binarized values of classes done using scikit-learn's LabelBinarizer.

- `p1flower.model` is the saved model after training. `classify.py` calls it when you run it.

- `plot.png` is the model accuracy and loss plots obtained after training.


## Working

I collected the dataset using Bing Image Search API. I learnt about that from [Here](https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/)

The Bing Image API really makes it quite easy to get small datasets. If you want to skip the above article here are the steps you will have to do to make your own dataset.

### How to collect your own datasets
> - Go to [Bing Image Search API](https://azure.microsoft.com/en-us/try/cognitive-services/?api=bing-image-search-api)
> - Scroll down a bit to and look for Bing Search API v7. Click on `Get API Key` and you will get options about Guest User, Free Azure Account and Sign In. You can proceed as per your suitability.
> - After you are done with registeration, going to [Bing Image Search API](https://azure.microsoft.com/en-us/try/cognitive-services/?api=bing-image-search-api) will show you your keys now.
> - You get two keys for `Bing Search API v7` and there are different endpoints.
> - We are going to use the endpoint which returns us images i.e. `https://api.cognitive.microsoft.com/bing/v7.0/images`
> - The code under [bing_search_api.py](https://api.cognitive.microsoft.com/bing/v7.0/images) deals with getting images by using the API keys
> - On line 21 of `bing_search_api.py` you need to paste your key at `YOUR_API_KEY = ` to get it working.
> - The usage is pretty simple
>   - Open terminal or cmd, and change to the directory where you want your images to be. Let's say `data`. Then run

        mkdir data/cascabela

        python search_bing_api.py --query "cascabela flower" --output dataset/cascabela

>>  All you images will be downloaded in that directory. This script collects 250 images for the inout you give.Some images have wrong formatting and the script deletes them automatically.

>> You need to type in the desired query after `--query` as shown above. Replace "\<cascabela flower>" with your desired input

>> You need to have the desired directory for saving images after `--output`

>>  A nice trick you can do is to go to bing search --> Image ; and then look up the images for the class you want, it will give you a fair idea about the dataset. I had to use these five classes because some others I researched had very poor images for example, tons of leaves and less of flowers and hands etc.

>> Run the above commands for the types of dataset you want. Make sure that you do not run out of requests as Bing only gives 3000 transactions per month for Image Search end point.

## Making Predictions

`classify.py` is used to make predicitions. Run it from the command line using

        python classify.py --model p1flower.model --labelbin lb.pickle --image examples/datura_1.jpg

(Windows users have to use `\\` instead of `/` )

- Note that, rename your input image as the desired class
 this becomes the ground truth for the prediction, it will mention if the prediction was correct or not. If you do not have the name of ground truth in the name of input image, prediction probability will be shown alongside `(incorrect)` even though prediction was correct.

The output you will get will be like this:

![](https://github.com/janmejai2002/Poisonous-Flower-Classification-using-Deep-Learning/blob/master/results/results.png)


Press `q` to exit the window or cmd+C or Ctrl+C in the terminal to stop execution.


Hope you liked my project. I have tested the dependencies in `requirements.txt` but if you run into errors, you can create an issue, I will be happy to help :D

The future scope for this project would be to

- Add more classes
- Turn it into an object detection model.


References I have used:
1. [How to build your own dataset](https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/)

2. [Keras and CNN](https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/)
