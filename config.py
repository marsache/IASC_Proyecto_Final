class Settings():
    ollama_model = "llama3:8b"
    temperature = 0.1

    base_dataset_path = "./datasets/"

    n_estimators = 200
    max_depth = 4
    learning_rate = 0.1
    random_state = 42


def getSettings() -> Settings:
    return Settings()