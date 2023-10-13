from config.config import CFG
from src.model.model import Model

if __name__ == '__main__':
    # Initialize model
    model = Model(CFG)

    # Load raw
    model.load_data()

    # Build model
    model.build()

    # Train model
    model.train()
