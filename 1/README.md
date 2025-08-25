## Example output

```python
python3 main.py
First response: Message(content=None, role='assistant', tool_calls=[ChatCompletionMessageToolCall(function=Function(arguments='{"location": "\\u0158evnice"}', name='get_location_coordinates'), id='chatcmpl-tool-6f69cb7a913f47f59115ff8f6f34ceff', type='function')], function_call=None, provider_specific_fields={'refusal': None})
Getting coordinates for Řevnice
{'location': 'Řevnice', 'latitude': 49.914333, 'longitude': 14.2361169}
Second response: Message(content='The longitude and latitude of Řevnice are 14.2361169 and 49.914333, respectively.', role='assistant', tool_calls=None, function_call=None, provider_specific_fields={'refusal': None})
--- Full response: ---
Message(content='The longitude and latitude of Řevnice are 14.2361169 and 49.914333, respectively.', role='assistant', tool_calls=None, function_call=None, provider_specific_fields={'refusal': None})
--- Response text: ---
The longitude and latitude of Řevnice are 14.2361169 and 49.914333, respectively.
(venv) ~/D/w/a/1 ❯❯❯ 
```