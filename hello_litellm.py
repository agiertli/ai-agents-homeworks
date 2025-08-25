import os
from dotenv import load_dotenv
from litellm import completion

# Load environment variables from .env file
load_dotenv()

# Simple hello world with LiteLLM
def main():
    try:
        # Create a simple completion request
        response = completion(
            model="openai/gpt-3.5-turbo",  # Using OpenAI model via LiteLLM
            messages=[
                {"role": "user", "content": "Say hello world in a friendly way!"}
            ],
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
        
        # Print the response
        print("LiteLLM Hello World Response:")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your .env file has OPENAI_API_KEY set correctly!")

if __name__ == "__main__":
    main()
