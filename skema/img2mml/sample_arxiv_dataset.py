from skema.img2mml.sampling_dataset.uniform_sampling import create_dataset
import time


if __name__ == "__main__":
    start_time = time.time()

    create_dataset()
    print("sampled_data is ready.")

    total_time = (time.time() - start_time) / 3600
    print(f"total time taken {total_time} hours.")
