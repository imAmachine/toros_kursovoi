from dotenv import load_dotenv
import os

load_dotenv(override=True)

ANALYSIS_OUTPUT_FOLDER_PATH = os.getenv('ANALYSIS_OUTPUT_FOLDER_PATH')