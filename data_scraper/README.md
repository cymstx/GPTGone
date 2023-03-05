# Instructions

## Setup
1. Create a virtual environment and activate it 

    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
2. Install the requirements

    ```
    pip install -r requirements.txt
    ```
3. Login to your openai account by running:
  
      ```
      chatgpt install
      ```

4. Change the dataset path in `scraper.py` to the path of the dataset you want to use.

5. Adjust the parameters in `scraper.py` to your liking.

    - `saving_frequency`: The amount of questions to ask before saving it.
    - `threshold`: The maximum amount errors before terminating the program.
    - `timeout`: The maximum amount of seconds before timeout.