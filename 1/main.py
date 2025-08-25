import litellm
import os
import json
import yfinance as yf
from pprint import pprint
from dotenv import load_dotenv
from geopy.geocoders import Nominatim

load_dotenv()  

model = "openai/"+os.environ.get("MODEL")


def get_location_coordinates(location: str):
    print(f"Getting coordinates for {location}")
    """Get latitude and longitude for a given location string."""
    geolocator = Nominatim(user_agent="location_app")
    try:
        location_data = geolocator.geocode(location)
        if location_data:
            return {
                "location": location,
                "latitude": location_data.latitude,
                "longitude": location_data.longitude
            }
        else:
            return {"location": location, "latitude": None, "longitude": None, "error": "Location not found"}
    except Exception as e:
        return {"location": location, "latitude": None, "longitude": None, "error": str(e)}


# Define custom tools
tools = [
     {
        "type": "function",
        "function": {
            "name": "get_location_coordinates",
            "description": "Use this function to get the Longitude and Latitude of a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location for which you want to get the coordinates, i.e. Prague",
                    }
                },
                "required": ["location"],
            },
        }
    }
]

available_functions = {
    "get_location_coordinates": get_location_coordinates
}

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the longitude and latitude of Řevnice?"},
]

def get_completion_from_messages(messages, model):

    response = litellm.completion(
        model=model,              
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    response_message = response.choices[0].message

    print("First response:", response_message)

    if response_message.tool_calls:
        # Find the tool call content
        tool_call = response_message.tool_calls[0]

        # Extract tool name and arguments
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments) 
        tool_id = tool_call.id
        
        # Call the function
        function_to_call = available_functions[function_name]
        function_response = function_to_call(**function_args)

        print(function_response)

        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_id,  
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(function_args),
                    }
                }
            ]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,  
            "name": function_name,
            "content": json.dumps(function_response),
        })

        # Second call to get final response based on function output
        second_response = litellm.completion(
            model=model,
            messages=messages,
            tools=tools,  
            tool_choice="auto"  
        )
        final_answer = second_response.choices[0].message

        print("Second response:", final_answer)
        return final_answer

    return "No relevant function call found."


response = get_completion_from_messages(messages, model)
print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
print(response.content)