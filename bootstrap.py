"""Recreate the deterministic demo dataset, train models, and evaluate."""
from simulation.generate_data import main as generate
from ml.train import main as train
from ml.evaluate import main as evaluate
if __name__ == "__main__":
    generate()
    train()
    evaluate()
