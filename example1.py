"""
Example 1: Define a VariableTempo object with a mathematical expression.

This script
    1) creates a ExpressionVariableTempo object using the expression
       '60 * (2 * t - 1) ** 2 + 60',
    2) prints the object information,
    3) prints the time in seconds for each of the first 40 beats, and
    4) graphs the tempo function.

"""
from variable_tempo import ExpressionVariableTempo


def example():
    """EXAMPLE 2: Construct a variable-tempo canon."""
    # Create expression tempo function.
    expr = "60 * (2 * t - 1) ** 2 + 60"
    vtf = ExpressionVariableTempo(expr=expr, length=1)
    print(vtf, "\n")

    # Find the corresponding time for the first 40 beats using the
    # beat_to_time method.
    print("Times for the first 40 beats:")
    for b in range(41):
        t = round(60 * vtf.beat_to_time(b), 3)
        print("Beat {} at {} seconds".format(b, t))

    vtf.graph()

if __name__ == "__main__":
    example()
