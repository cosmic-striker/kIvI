import os

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your_secret_key')
    DEBUG = False
    TESTING = False
    MODEL_PATH = os.environ.get('MODEL_PATH', 'model/llama2/')
    # Add more general configurations here

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    # Add development-specific configurations here

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    # Add testing-specific configurations here

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    # Add production-specific configurations here

def get_config(env='development'):
    """Retrieve the configuration based on the environment."""
    if env == 'development':
        return DevelopmentConfig
    elif env == 'testing':
        return TestingConfig
    elif env == 'production':
        return ProductionConfig
    else:
        raise ValueError(f"Unknown environment: {env}")
