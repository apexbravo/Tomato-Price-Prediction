import os
import pickle
from django.conf import settings
from django.http import HttpResponse, JsonResponse
import json
from django.shortcuts import render
from keras_preprocessing.image import load_img

from keras_preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
# Create your views here.

with open('C:\\Users\\apexb\\Documents\\Machine Learning\\Gunde\\Tomato Price Prediction\\mysite\\tomatop\\model_pkl', 'rb') as f:
    lin_reg_model = pickle.load(f)


def index(request) -> HttpResponse:
    return render(request, 'tomatop/index.html')


def predictImage(request) -> HttpResponse:
    return render(request, 'tomatop/pages/predictImage.html')


def predictPrice(request) -> HttpResponse:
    return render(request, 'tomatop/pages/predictPrice.html')


model = load_model(
    'C:\\Users\\apexb\\Documents\\Machine Learning\\Gunde\\Tomato Price Prediction\\mysite\\tomatop\\my_model.h5')


def predict(request):
    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_image = request.FILES['tomato-image']

        # Save the uploaded image to a temporary file
        img_file_path = os.path.join(
            settings.BASE_DIR, 'tmp', uploaded_image.name)
        with open(img_file_path, 'wb') as f:
            for chunk in uploaded_image.chunks():
                f.write(chunk)

        # Load the image using Keras preprocessing
        img = load_img(img_file_path, target_size=(120, 120))
        x = img_to_array(img)
        x = preprocess_input(x)

        # Make the prediction
        predicted_price = model.predict(np.expand_dims(x, axis=0))[0][0]

        # Return the predicted price to the template
        return JsonResponse({'price': float(predicted_price)})
    else:
        # Return the empty form to the template
        return render(request, 'index.html')


def preprocess_input(x):
    x = x.astype('float32')
    x /= 255.0
    return x


def predict_price(request):
    if request.method == 'POST':
        # get the input values from the form
        tomato_type = request.POST.get('tomato_type')
        month = request.POST.get('month')
        season = request.POST.get('season')
        quantity = request.POST.get('quantity')
        size = request.POST.get('size')
        ripeness = request.POST.get('ripeness')
        insects_diseases = request.POST.get('insects_diseases')

        # replace the categorical values with numerical values
        tomato_type = {'Trinity': 0, 'Star9': 1, 'Star6': 2,
                       'RomaVF': 3, 'Daisy': 4, 'Supersweet': 5}[tomato_type]
        month = {'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5, 'July': 6,
                 'August': 7, 'September': 8, 'October': 9, 'November': 10, 'December': 11}[month]
        season = {'Summer': 0, 'Winter': 1, 'CNG': 2}[season]
        quantity = {'L': 0, 'M': 1, 'S': 2}[quantity]
        size = {'Small': 0, 'Mid': 1, 'Big': 2}[size]
        ripeness = {'G': 0, 'B': 1, 'T': 2, 'P': 3, 'R': 4}[ripeness]
        insects_diseases = {'Y': 1, 'N': 0}[insects_diseases]

        # create a list of input values
        input_data = [[tomato_type, month, season,
                       quantity, size, ripeness, insects_diseases]]

        # predict the price using the loaded model
        predicted_price = lin_reg_model.predict(input_data)[0]

        # display the predicted price
        return render(request, 'predictPrice.html', {'predicted_price': predicted_price})

    else:
        # display the input form
        return render(request, 'input_form.html')
