# Setup Instructions

## Environment Setup

1. **Virtual Environment** (already created)
   - Virtual environment is located in `venv/`
   - To activate it, run: `source venv/bin/activate`

2. **Dependencies** (already installed)
   - All required packages are installed via `requirements.txt`

3. **Environment Variables**
   Create a `.env` file in the project root with the following content:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   BASE_URL=https://api.openai.com/v1
   ```

## Running the Application

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Make sure your `.env` file has the correct OpenAI API key

3. Run the original script:
   ```bash
   python3 1/main.python
   ```

4. Or run the simple LiteLLM hello world:
   ```bash
   python3 hello_litellm.py
   ```

## Project Structure

- `1/main.python` - Main application script that uses OpenAI API to get stock information
- `hello_litellm.py` - Simple LiteLLM hello world example
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules for Python projects
- `venv/` - Virtual environment (ignored by git)
