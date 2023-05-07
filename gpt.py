import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the conversation loop
while True:
    # Get user input
    user_input = input("You: ")

    # Encode the user input
    encoded_input = tokenizer.encode(user_input, return_tensors="pt").to(device)

    # Generate a response from the model
    response = model.generate(encoded_input, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, top_p=0.95, top_k=50, temperature=0.7)

    # Decode the response
    decoded_response = tokenizer.decode(response[0], skip_special_tokens=True)

    # Print the response
    print("ChatGPT:", decoded_response)



sample_outputs = model.generate(generated, 
                                do_sample=True, 
                                max_length=50, 
                                top_p=0.95, 
                                top_k=50, 
                                temperature=0.7)
