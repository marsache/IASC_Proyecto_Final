class Settings():
    ollama_model = "llama3:8b"
    temperature = 0.1

    base_dataset_path = "./datasets/"


def getSettings() -> Settings:
    return Settings()