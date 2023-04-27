from skema.img2mml.sampling_dataset.without_mp import main
import sys, time

if __name__ == "__main__":

    start_time = time.time()

    main()
    print("sampled_data is ready.")

    total_time = (time.time() - start_time) / 3600
    print(f"total time taken {total_time} hours.")
