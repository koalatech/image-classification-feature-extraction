import pickle
from img2vec_pytorch import Img2Vec
from PIL import Image


with open('./model.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

image_path = './example/rain.jpeg'

img = Image.open(image_path)

features = img2vec.get_vec(img)

pred = model.predict([features])

print(pred)