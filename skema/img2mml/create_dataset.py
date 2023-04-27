from skema.img2mml.sampling_dataset.sampling_with_mp import create_dataset
import sys, time
import argparse

# Define the command-line arguments
parser = argparse.ArgumentParser(description="Create dataset for training")
parser.add_argument(
    "--dataset",
    choices=["im2mml", "arxiv"],
    default="arxiv",
    help="Choose one dataset (im2mml or arxiv) to create",
)

# Parse the command-line arguments
args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()

    create_dataset(args.dataset)
    print("sampled_data is ready.")

    total_time = (time.time() - start_time) / 3600
    print(f"total time taken {total_time} hours.")
