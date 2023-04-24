from skema.img2mml.sampling_dataset.temp import main
import sys, time

if __name__ == "__main__":

    start_time = time.time()

    main()
    print("sampled_data is ready.")
    sys.exit()

    total_time = (time.time() - start_time) / 3600
    print(f"total time taken {total_time} hours.")
