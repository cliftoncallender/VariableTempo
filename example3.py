"""
Example 3: Nancarrow geometric acceleration.

Create a variable tempo object based on Nancarrow's geometric acceleration,
and print the corresponding time in seconds for the first six beats.

"""
from variable_tempo import NancarrowGeometricAcceleration


def example():
    """Construct a variable-tempo canon."""
    nancarrow = NancarrowGeometricAcceleration(initial_duration=2, percent=5)
    print(nancarrow, "\n")
    for beat in range(7):
        time = round(nancarrow.beat_to_time(beat) * 60, 3)
        print("Beat {} at {} seconds".format(beat, time))  # 0, 2, 3.905, ...


if __name__ == "__main__":
    example()
