# %%
import time
from synthesis.blinkfill import learn, run


def main():
    # Example
    inputs = [
        "Mumbai, India",
        "Los Angeles, United States of America",
        "Newark, United States",
        "New York, United States of America",
        "Wellington, New Zealand",
        "New Delhi, India",
    ]
    outputs = [
        "India",
        "United States of America",
    ]

    start = time.perf_counter()
    program = learn(inputs, outputs)
    learn_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    result = run(program, "Newark, United States")
    run_elapsed = time.perf_counter() - start

    print(program)
    print(f'Output: "{result}"')
    print(f"Learn took: {learn_elapsed:0.2f}s")
    print(f"Run took: {run_elapsed:0.2f}s")


if __name__ == "__main__":
    main()
