import base64
from io import BytesIO
import os
import pickle
from django.conf import settings
from django.http import HttpResponse, JsonResponse
import json
from django.shortcuts import render
from keras_preprocessing.image import load_img
import locale
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


from PIL import Image

def predict(request):
    if request.method == 'POST':
        if 'tomato-image' in request.FILES:
            # Get the uploaded image file
            uploaded_image = request.FILES['tomato-image']
        else:
            # Capture the image from webcam
            data_uri = request.POST.get('image_data', None)
            if data_uri:
                # Extract the base64 encoded image data
                image_data = data_uri.split(',')[1]
                # Convert the base64 encoded image data to bytes
                image_bytes = base64.b64decode(image_data)
                # Create a BytesIO object from the image bytes
                uploaded_image = BytesIO(image_bytes)
            else:
                return JsonResponse({'error': 'No image data found.'})

        # Load and preprocess the image using PIL (Python Imaging Library)
        img = Image.open(uploaded_image).convert('RGB')
        img = img.resize((120, 120))
        x = np.array(img)
        x = preprocess_input(x)

        # Make the prediction
        predicted_class = model.predict(np.expand_dims(x, axis=0))[0][0]

        # Check if the predicted class is 'tomato' with a threshold of 0.7
        if predicted_class >= 70.0:
            # The image is a tomato, make the price prediction
            predicted_price = model.predict(np.expand_dims(x, axis=0))[0][0]
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

            # Divide the value by 100 to convert it from cents to dollars
            predicted_price_in_dollars = float(predicted_price) / 100

            # Format the value using the user's locale
            formatted_price = locale.currency(predicted_price_in_dollars)

            # Return the formatted value as a JSON response
            return JsonResponse({'price': formatted_price})
        else:
            # The image is not a tomato, return an error message
            return JsonResponse({'error': 'The uploaded image is not a tomato.'})

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
        # format the predicted price as a string in the format of $x.xx
        formatted_price = '{:,.2f}'.format(predicted_price)

        # display the predicted price
        return JsonResponse({'price': formatted_price})

    else:
        # display the input form
        return render(request, 'input_form.html')
