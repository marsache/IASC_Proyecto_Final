class Settings():
    ollama_model = "llama3:8b"
    temperature = 0.1


def getSettings() -> Settings:
    return Settings()