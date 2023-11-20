from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse, HttpResponse
#from django.views.decorators.csrf import csrf_exempt
#from .models import InsuraTransformer, Lang
#from .key_map import key_mapping
import torch #torch version 2.1.0+cpu
#import torch.nn as nn
#import torch.optim as optim
import json
import os
import gpt_2_simple as gpt2
import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import copy

# Create your views here.

def we_do_page(request):
    return render(request, 'we_do.html')

def products_page(request):
    return render(request, 'products.html')

def claims_services_page(request):
    return render(request, 'claims.html')

def work_with_us_page(request):
    return render(request, 'work.html')

def about_us_page(request):
    return render(request, 'about.html')


device = torch.device('cpu')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
def chat(request):
    return render(request, 'chat.html')

def infer(model, input):
    input = "<SOS> " + input + " <BOT>: "
    input = tokenizer(input, return_tensors='pt')
    X = input['input_ids'].to(device)
    attn_mask = input['attention_mask'].to(device)
    output = model.generate(X, 
                            attention_mask=attn_mask, 
                            max_length=150, 
                            num_return_sequences=1,
                            pad_token_id=model.config.eos_token_id,
                            top_k=50,
                            #top_p=0.95) 
    )
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

def ai_response(request):
    try:
        data = json.loads(request.body)
        user_message = data.get('userMessage', '')
        print(f"User Message: {user_message}")
        
    except json.JSONDecodeError:
        # Handle JSON decoding error
        response = {'reply': 'Invalid JSON data'}
        return JsonResponse(response, status=400)
    
    model_path = 'model/gpt2_qa_bot.pth'
    #model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpt2_qa_bot.pth')
    

    # Check if the model file exists
    if not os.path.exists(model_path):
        response = {'reply': 'No Model found in the directory'}
        return JsonResponse(response, status=404)
    else:
        try: 
            gpt2model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
            gpt2model.resize_token_embeddings(len(tokenizer))

            # Load the state dictionary
            state_dict = torch.load(model_path, map_location=device)
            model_response = infer(gpt2model, user_message)
            
            
            # Extract the first response from model_response
            first_response = model_response.split('<BOT>:')[1].strip().replace('-', '').replace('~', '')
            #first_response = first_response.replace('-', '')
            response = '<BOT>' + first_response
            print(f"Model Response: {response}")
            return HttpResponse(response)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_response = {'reply': f'Error: {str(e)}'}
            return JsonResponse(error_response, status=500)
