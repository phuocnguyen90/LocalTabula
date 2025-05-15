
from utils.llm_interface import LLMWrapper

import os
from dotenv import load_dotenv
from utils.resources import initialize_all_models
load_dotenv('config/.env')

llm_wrapper, aux_models =initialize_all_models()
llm_wrapper.generate_response("Hello, how are you?")