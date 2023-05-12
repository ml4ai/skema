from skema.img2mml.sampling_dataset.sampling_dataset import main
import sys, time

if __name__ == "__main__":

    start_time = time.time()

    main()
    print("sampled_data is ready.")

    total_time = (time.time() - start_time) / 60
    print(f"total time taken {total_time} minutes.")
