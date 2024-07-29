import os
from dotenv import load_dotenv

class KatzBotConfig:
    """
    Configuration class for KatzBot application.

    This class is responsible for loading environment variables, setting up
    base directories, and providing paths for required files and API keys.
    """

    def __init__(self):
        """
        Initialize the configuration with the given base directory.

        Args:
            base_dir (str): The base directory path where the application resides.
        """
        
        # self.BASE_DIR = r''
        self.load_environment_variables()
        self.setup_file_paths()
        self.load_api_keys()

    def load_environment_variables(self):
        """Load environment variables from a .env file."""
        load_dotenv()

    def setup_file_paths(self):
        """Setup file paths for various resources required by the application."""
        self.BOT_IMAGE_PATH = r'static/static/ai_icon.png'
        self.HUMAN_IMAGE_PATH = r'static/static/user_icon.png'
        self.CSS_PATH = r'static/static/styles.css'
        # self.DATA_FILE = os.path.join(self.BASE_DIR, r'data/loaded_data_train_test_2.pkl')
        self.DATA_FILE = r'data/loaded_dataset_website.pkl'
        self.CHAT_HISTORY_FILE = r'session_chat_history.csv'
        
        # self.model_choice=["Mixtral-8x7b-32768", "Gemma2-9b-It", "Gemma-7b-It", "Llama-3.1-70b-Versatile", "Llama-3.1-8b-Instant", "Llama3-70b-8192", "Llama3-8b-8192"]
        # self.model_name = self.model_choice[0]
        
        self.models_choices = {
                                1: "Mixtral-8x7b-32768",
                                2: "Gemma2-9b-It",
                                3: "Gemma-7b-It",
                                4: "Llama-3.1-70b-Versatile",
                                5: "Llama-3.1-8b-Instant",
                                6: "Llama3-70b-8192",
                                7: "Llama3-8b-8192"
                            }

        self.model_name = self.models_choices[1]

    def load_api_keys(self):
        """Load API keys from environment variables."""
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


if __name__ == '__main__':
    config = KatzBotConfig()


