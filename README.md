<!-- Title -->

# BERT model for Cyberbulling Classification
<br>

<!-- TABLE OF CONTENTS -->

<details open="open">
  <summary><h1 style="display: inline-block">Table of Contents</h1></summary>
  <ol>
    <li>
        <a href="#how-to-use-the-model">How to use the model</a>
        <ul>
         <li><a href="#requirements">Requirements</a></li>
         <li><a href="#how-to-load-the-model">How to load the model</a></li>
         <li><a href="#processing-the-input">Processing the input</a></li>
         <li><a href="#prediction">Prediction</a></li>
      </ul>
    </li>
    <li><a href="#models-evaluation">Model's evaluation</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<br>

<!-- Introduction -->
# **How to use the model**
The model in the repo is a BERT model that has been trained to make binary classifications of text/tweet in order to detect cyberbullying behaviours.

The following sections will explain briefly how to gain access to and use the model in Python.
## **Requirements**
To begin, the following packages are required to run the model:
- *tensorflow_addons*
- *transformers*

And they can be installed using *pip* with the following commands:
```
    pip install tensorflow_addons
    pip install transformers
```

## **How to load the model**
We can now load the pretrained model after installing the previous packages.
The model file is stored in the project's `bert_tweet/models/` directory. 

The following snippet shows an example of how load it:

```python
    # Import of the needed libraries
    import tensorflow as tf
    from tensorflow_addons.optimizers import AdamW
    from transformers import AutoTokenizer,TFRobertaModel
    import transformers

    # Libraries not needed for the loading, but useful for the next steps
    import pandas as pd
    import numpy as np

    # Load of the model
    model = tf.keras.models.load_model('MODEL_PATH/model32_0.35_1e-05_5e-07.h5',custom_objects={"TFRobertaModel": transformers.TFRobertaModel})
```

## **Processing the input**
We'd like to use the model after loading it, but we can leave the data in input as is. As a result, we must process them before passing them to the model, specifically tokenizing them. We can use the `AutoTokenizer` for this purpose, as shown in the following snippet:

```python
string = "UNISA will rule the world, and this is a promise. Let's kill other university students and make them our slaves."

#formatting text to avoid characters not included in the encoding dictionary of the tokenizer
x = (string
     .lower()     
     .replace('\x89Ûª|Ûª', "'")
     .replace('\n|\x89.|\x9d *', ' ')
     .replace('&gt;', ">")
     .replace('&lt;', "<")
     .replace('&amp;', " and ")
     .replace('won\'t', 'will not')
     .replace('can\'t', 'cannot')
     .replace('i\'m', 'i am')
     .replace('ain\'t', 'is not')
     .replace('hwy.', 'highway')
     .replace('(\w+)\'ll', '\g<1> will')
     .replace('(\w+)n\'t', '\g<1> not')
     .replace('(\w+)\'ve', '\g<1> have')
     .replace('(\w+)\'s', '\g<1> is')
     .replace('(\w+)\'re', '\g<1> are')
     .replace('(\w+)\'d', '\g<1> would')     
     .replace('(\w+)\'m', '\g<1> am')
     .replace('<3', 'love')
     .replace('w/e', 'whatever')
     .replace('w/', 'with')    
     .replace('\b', ' ')
     .replace('-', ' ')
     .replace('  *', ' ')
    )

#let's tokenize the text
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', 
                        normalization=True, 
                        use_fast = False,
                        add_special_tokens=True,
                        pad_to_max_length=True,
                        return_attention_mask=True)
token = tokenizer(string, 
                  padding="max_length", 
                  truncation=True,
                  return_tensors = 'tf').data

```
The `token` can now be fed into the model as input to predict. Please keep in mind that the BERT model has a maximum length of 512 characters, so any characters that exceed the maximum will be truncated.

## **Prediction**
Finally, we can use the model by calling the `predict` method and passing the previously defined token. This method will generate an object in which we can find a real value greater than 0.5 as a result of the prediction if it is classified as cyberbullying behaviour, otherwise if it is not.

```python
#let's predict
prediction = model.predict(token)
# see the prediction, if > 0.5, it's a hate speech
print("Text:", string, "Prediction:", prediction[0][0])
```

# **Model's evaluation**
In this section, we will show a table that summarizes the model's evaluation. Various models have been trained in order to tune the hyperparameter, but we will only present the best one in the table below:
|Accuracy|F1|Precision|Recall|
|---|---|---|---|
|0.93|0.92|0.91|0.93|


<!-- LICENSE -->

# **License**

Distributed under the MIT License. See `LICENSE` for more information.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
