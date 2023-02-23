from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import pandas as pd

def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []  # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,  # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,  # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
                                                     input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN):
    train_InputExamples = train.apply(
        lambda x: InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this case
                               text_a=x[DATA_COLUMN],
                               text_b=None,
                               label=x[LABEL_COLUMN]), axis=1)

    validation_InputExamples = test.apply(
        lambda x: InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this case
                               text_a=x[DATA_COLUMN],
                               text_b=None,
                               label=x[LABEL_COLUMN]), axis=1)

    return train_InputExamples, validation_InputExamples


def model_conversion(train,test):
    #model definition and loading bert model
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model.summary()

    for i in train.take(1):
      #text
      train_feat = i[0].numpy()
      #label
      train_lab = i[1].numpy()

    train = pd.DataFrame([train_feat, train_lab]).T
    train.columns = ['DATA_COLUMN', 'LABEL_COLUMN']
    train['DATA_COLUMN'] = train['DATA_COLUMN'].str.decode("utf-8")
    train.head()

    for j in test.take(1):
      test_feat = j[0].numpy()
      test_lab = j[1].numpy()

    test = pd.DataFrame([test_feat, test_lab]).T
    test.columns = ['DATA_COLUMN', 'LABEL_COLUMN']
    test['DATA_COLUMN'] = test['DATA_COLUMN'].str.decode("utf-8")
    test.head()

    DATA_COLUMN = 'DATA_COLUMN'
    LABEL_COLUMN = 'LABEL_COLUMN'
    train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

    train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
    train_data = train_data.shuffle(100).batch(16).repeat(2)

    validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
    validation_data = validation_data.batch(16)

    return train_data, validation_data, model


def model_training(train_data,validation_data,model):

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
                  metrics=['accuracy'])

    model.fit(train_data, epochs=10, validation_data=validation_data)

    return model

def model_saving(model):
    model.save_pretrained('bert_model')

def model_loading():

    # train and test dataset
    data = pd.read_csv('../merging/dataset.csv')
    #take only observations with class label non empty
    data = data[data['class_label'].notna()]

    #split dataset in each class label
    data_0 = data[data['class_label'] == 0]
    data_1 = data[data['class_label'] == 1]

    #setting 80% of data for training and 20% for testing, half for each class label
    train_0 = data_0.sample(frac=0.8, random_state=0)
    test_0 = data_0.drop(train_0.index)
    train_1 = data_1.sample(frac=0.8, random_state=0)
    test_1 = data_1.drop(train_1.index)

    # merge train and test dataset
    train = pd.concat([train_0, train_1])
    test = pd.concat([test_0, test_1])
    #convert to tensors
    train = tf.data.Dataset.from_tensor_slices((train['Text'].values, train['class_label'].values))
    test = tf.data.Dataset.from_tensor_slices((test['Text'].values, test['class_label'].values))
    return train, test

def model_pipeline():
    train, test = model_loading()
    train_data, validation_data, model = model_conversion(train, test)
    model = model_training(train_data, validation_data, model)
    model_saving(model)


def main():
    model_pipeline()

if __name__ == "__main__":
    main()