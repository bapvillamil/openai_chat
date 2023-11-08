from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
import torch
import torch.nn as nn
import json
import os


# Create your views here.
def chat(request):
    return render(request, 'chat.html')

def ai_response(request):
    user_message = request.POST.get('userMessage', '')

    # Define the model file path
    model_path = 'ai_model.pth'  # Adjust the path accordingly

    # Check if the model file exists
    if not os.path.exists(model_path):
        response = {'reply': 'No Model found in the directory'}
    else:
        # Load the model
        model = torch.load(model_path, map_location='cpu')
        model.eval()

        # Process the user's message and generate an AI response
        with torch.no_grad():
            response = model(user_message)

        response = {'reply': response}  # Modify this to format the AI response as needed

    return JsonResponse(response)