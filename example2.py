"""
Example 2: Construct a variable-tempo canon.

    1) Create an exponential tempo function
    2) Creat break-point function
    3) Use music21 to read musicxml theme
    4) Transform theme with the tempo functions
    5) Construct canon
    6) Save MIDI files

"""
from variable_tempo import ExponentialVariableTempo, VariableTempoBPF
import music21 as m21


def example():
    """Construct a variable-tempo canon."""
    # Create variable tempo function.
    tempo_curve = ExponentialVariableTempo(start_tempo=60, end_tempo=120,
                                           num_beats=6)
    print(tempo_curve, "\n")

    # Create variable tempo break-point function
    # and add three copies of tempo_curve.
    bpf = VariableTempoBPF([tempo_curve for _ in range(4)])
    print(bpf)
    bpf.graph()

    # Parse theme, add tempo marking and show.
    themeURL = 'http://cliftoncallender.com/resources/theme.xml'
    theme = m21.converter.parse(themeURL)
    theme.insert(0, m21.tempo.MetronomeMark(number=60))
    theme.show()

    # Transform theme by tempo function.
    transformed_theme = theme.variableAugmentAndDiminish(bpf)
    transformed_theme.show('t')
    # transformed_theme.show() creates problems with very small durations:
    # "music21.duration.DurationException: Cannot return types smaller than
    # 2048th"

    # Create score of the theme in a tempo canon.
    canon = m21.stream.Score()
    # Imitition occurs every six beats
    imitative_distance = tempo_curve.beat_to_time(6) * 60
    delays = [imitative_distance * i for i in range(3)]
    transpositions = [-24, -2, 22]
    for delay, transposition in zip(delays, transpositions):
        canon.insert(delay, transformed_theme.transpose(transposition))

    # Save midi files of theme, transformed theme, and canon for comparison.
    theme.write('midi', fp='theme.midi')
    transformed_theme.write('midi', fp='transformed_theme.midi')
    canon.write('midi', fp='canon.midi')


if __name__ == "__main__":
    example()
