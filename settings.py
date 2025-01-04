from dotenv import load_dotenv
import os

load_dotenv(override=True)

MASKS_FOLDER_PATH = os.getenv('MASKS_FOLDER_PATH')
SOURCE_IMAGES_FOLDER_PATH = os.getenv('SOURCE_IMAGES_FOLDER_PATH')
ANALYSIS_OUTPUT_FOLDER_PATH = os.getenv('ANALYSIS_OUTPUT_FOLDER_PATH')