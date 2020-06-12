import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.models import Model
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
import cv2
from keras.preprocessing.sequence import pad_sequences
model_path = 'final_model.h5'
with open("vocab_file.txt", 'r') as f:
    vocabulary_list = [line.rstrip('\n') for line in f]

word_to_index = {}
index_to_word = {}

index=1
for word in vocabulary_list:
    word_to_index[word]=index
    index_to_word[index]=word
    index=index+1

print(len(word_to_index))

def caption(photo,max_length_of_caption):
    caption_model = load_model('final_model.h5')
    in_text = 'start_caption '
    for i in range(max_length_of_caption):
        sequence = [word_to_index[w] for w in in_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length_of_caption)
        yhat = caption_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_word[yhat]
        in_text += ' ' + word
        if word == 'end_caption':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def gen_caption(file):
    model_inception = load_model('inception_V3_model.h5')
    model_inception = Model(model_inception.input, model_inception.layers[-2].output)
    my_id_imgarr_dict = {}
    from keras.preprocessing import image
    img = image.load_img(file, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model_inception.predict(x)
    x = np.reshape(predictions, predictions.shape[1])
    my_id_imgarr_dict[file] = x
    image = x.reshape((1, 2048))
    caption_gen = caption(image, 36)
    print("Caption for image :", caption_gen)
    return  caption_gen