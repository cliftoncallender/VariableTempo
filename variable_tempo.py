"""Classes for working with continuous variable tempo functions."""

import music21 as m21
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import namedtuple
import copy


class VariableTempoBase:
    """Base class for continuously variable tempos.

    Base class contains a `graph` method to be inherited by its subclasses,
    `VariableTempo` and `VariableTempoBPF`.
    """

    def graph(self, title=None):
        """Graph tempo function using maplotlib.pyplot."""
        if not self.length:
            self.length = 1
        step = self.length / 100
        times = np.arange(0., self.length + step, step)
        tempos = [self.instantaneous_tempo(time) for time in times]
        plt.xlabel("time (minutes)")
        plt.ylabel("tempo (beats per minute)")
        plt.plot(times, tempos)
        if title:
            plt.title(title)
        plt.show()


class VariableTempo(VariableTempoBase):
    """Class for handling continuously variable tempos.

    Class contains methods for converting between time and beats, as well as
    getting the instantaneous tempo at a given time.

    Attributes:
        curve_type (str): `curve_type` can be 'constant', 'linear',
            'exponential', or 'expression'

        start_tempo (float): Initial tempo for the variable tempo function.
            Defaults to `None`. `start_tempo` is assumed to be in beats per
            minute and is required for 'constant', 'linear', and 'exponential'
            tempo functions.

        end_tempo (float): Ending tempo for the variable tempo function.
            Defaults to `None`. `end_tempo` is assumed to be in beats per
            minute and is required attribute for 'linear' and 'exponential'
            tempo functions.

        length (float): Length of time (in minutes) from `start_tempo` to
            `end_tempo`. Defaults to `None`. 'linear' and 'exponential' tempo
            functions require a value for either `length` or `num_beats` (see
            below). If a value for `length` is not provided, it is derived
            from `num_beats`.

        num_beats (float): Number of beats occuring between `start_tempo` and
            `end_tempo`. Defaults to `None`. 'linear' and 'exponential' tempo
            functions require a value for either `num_beats` or `length` (see
            above). If a value for `num_beats` is not provided, it is derived
            from `length`.

        expr (str): `expr` must be a string that gives rise to a valid
            mathematical expression of the single variable 't' when evaluated
            by the `eval` function. Tempo functions defined by expressions
            depend on the `quad` function in `scipy.integrate`.

    Examples:
        Create a `VariableTempo` object for a constant tempo of 60 beats per
        minute--f(t) = 60:

            >>> curve = VariableTempo('constant', start_tempo=60)


        Create a `VariableTempo` object for a linear acceleration from 60 bpm
        to 120 bpm over 0.5 minutes--f(t) = 60 * 2 * t + 60:

            >>> curve = VariableTempo('linear', start_tempo=60,
            ...                       end_tempo=120, length=0.5)


        Create a `VariableTempo` object for an exponential acceleration from 60
        bpm to 120 bpm over 2 minutes--f(t) = 60 + (120/60)^(t/2):

            >>> curve = VariableTempo('exponential', start_tempo=60,
            ...                       end_tempo=120, length=2)


        Create a `VariableTempo` object for a tempo of the form
        f(t) = 60t^2 + 60:

            >>> curve = VariableTempo('expression', expr='60 * t**2 + 60')

    """

    def __init__(self, curve_type, start_tempo=None, end_tempo=None,
                 length=None, num_beats=None, expr=None):
        """Initialize object depending on curve_type."""
        self.curve_type = curve_type
        self.start_tempo = start_tempo
        self.end_tempo = end_tempo
        self.length = length
        self.num_beats = num_beats

        # Initialize based on curve_type.
        if curve_type is 'constant':
            self._constant_init()
        elif curve_type is 'linear':
            self._linear_init()
        elif curve_type is 'exponential':
            self._exponential_init()
        elif curve_type is 'expression':
            self._expression_init(expr)
        # Otherwise, curve_type is invalid.
        else:
            curve_types = ['constant', 'linear', 'exponential', 'expression']
            raise Exception("curve_type must be " +
                            ', '.join(item for item in curve_types) + '.')

    def _constant_init(self):
        # Set expr.
        self.expr = "{}".format(self.start_tempo)

        # Set length if num_beats is not None.
        if self.num_beats:
            self.length = self.num_beats / self.start_tempo

    def _linear_init(self):
        # Check that length or num_beats is not None.
        self._check_length_or_num_beats()

        # Derive num_beats from length or length from num_beats.
        average_tempo = (self.start_tempo + self.end_tempo) / 2
        if self.length:
            self.num_beats = average_tempo * self.length
        else:
            self.length = self.num_beats / average_tempo

        # Set slope and expression.
        self.slope = (self.end_tempo - self.start_tempo) / self.length
        self.expr = self.expr = "{} * t + {}".format(self.slope,
                                                     self.start_tempo)

    def _exponential_init(self):
        # Check that length or num_beats is not None.
        self._check_length_or_num_beats()

        # Set ratio between end and start tempos.
        self.ratio = self.end_tempo / self.start_tempo

        # Derive num_beats from length or length from num_beats.
        conversion = self.start_tempo * (self.ratio - 1) / math.log(self.ratio)
        if self.length:
            self.num_beats = self.length * conversion
        else:
            self.length = self.num_beats / conversion

        # Set expression.
        self.expr = "{} * {}**(t / {})".format(self.start_tempo, self.ratio,
                                               self.length)

    def _expression_init(self, expr):
        # Set expression if expr is valid.
        if expr is not None:
            # test if expr can be evaluated by eval
            try:
                t = 1
                eval(expr)
                self.expr = expr
            except:
                raise ValueError("Invalid expression for expr")
        else:
            raise ValueError("For curve_type 'expression', expr cannot "
                             "be None")

    def _check_length_or_num_beats(self):
        # Check that 'linear' and 'exponential' functions have a value other
        # than None for either length or num_beats
        if not (self.length or self.num_beats):
            raise ValueError("'linear' and 'exponential' functions "
                             "require a value for either length or "
                             "num_beats.")

    def time_to_beat(self, t):
        """Given a time in minutes, returns the corresponding beat.

        The beat is the definite integral of the tempo function from 0 to `t`.

        Example:
            >>> curve = VariableTempo('linear', start_tempo=60,
            ...                       end_tempo=120, length=1)
            >>> curve.time_to_beat(0.5)
            37.5
        """
        if self.curve_type is 'constant':
            return self._constant_t2b(t)
        if self.curve_type is 'linear':
            return self._linear_t2b(t)
        if self.curve_type is 'exponential':
            return self._exponential_t2b(t)
        if self.curve_type is 'expression':
            return self._expression_t2b(t)

    # helper methods for .time_to_beat
    def _constant_t2b(self, t):
        return self.start_tempo * t

    def _linear_t2b(self, t):
        return self.slope * t**2 / 2 + self.start_tempo * t

    def _exponential_t2b(self, t):
        a = self.start_tempo
        r = self.ratio
        L = self.length
        return (a * L / math.log(r)) * (r**(t / L) - 1)

    def _expression_t2b(self, t):
        """Return the result of numerical integration with
        `scipy.integrate.quad`.
        """
        f = lambda t: eval(self.expr)
        return scipy.integrate.quad(f, 0, t)[0]

    def beat_to_time(self, beat):
        """Given a beat, returns the corresponding time in minutes.

        Inverse function of .time_to_beat. The corresponding time is determined
        by setting the definite integral of f(t) from 0 to t_0 = b and solving
        for t_0.

        Example:
            >>> curve = VariableTempo('linear', start_tempo=60,
            ...                       end_tempo=120, length=1)
            >>> curve.beat_time_tempo(37.5)
            0.5
        """
        if self.curve_type is 'constant':
            return self._constant_b2t(beat)
        if self.curve_type is 'linear':
            return self._linear_b2t(beat)
        if self.curve_type is 'exponential':
            return self._exponential_b2t(beat)
        if self.curve_type is 'expression':
            return self._expression_b2t(beat)

    # helper methods for .beat_to_time
    def _constant_b2t(self, beat):
        return beat / self.start_tempo

    def _linear_b2t(self, beat):
        a = self.slope / 2
        b = self.start_tempo
        c = -beat
        return (-b + (b**2 - 4 * a * c)**0.5) / (2 * a)

    def _exponential_b2t(self, beat):
        # Equation can yield ``ValueError: math domain error`` for beats
        # below a certain value. Need to determine exact value given a, r, L.
        a = self.start_tempo
        r = self.ratio
        L = self.length
        b = beat
        return L * math.log((b * math.log(r)) / (a * L) + 1, r)

    def _expression_b2t(self, beat):
        """Binary search using the inverse method."""
        max_error = 0.0001
        min_time = 0
        max_time = 0
        max_beat = 0

        while max_beat < beat:
            min_time = max_time
            max_time += 1
            max_beat = self._expression_t2b(max_time)
        beat_error = 1
        while beat_error > max_error:
            mid_time = (min_time + max_time) / 2.0
            mid_beat = self._expression_t2b(mid_time)
            if mid_beat < beat:
                min_time = mid_time
            else:
                max_time = mid_time
            beat_error = abs(mid_beat - beat)
        return mid_time

    def instantaneous_tempo(self, t):
        """Return instantaneous tempo at time t."""
        return eval(self.expr)

    def __repr__(self):
        string = "VariableTempo object:\n"
        string += "{} function ".format(self.curve_type)
        if self.curve_type is 'expression':
            string += self.expr
        elif self.curve_type is 'constant':
            string += "tempo = {}".format(self.start_tempo)
        else:
            string += ("from tempo = {} to {} over {} beats ({} seconds)"
                       .format(self.start_tempo, self.end_tempo,
                               self.num_beats, round(self.length * 60, 3)))
        return string


class VariableTempoBPF(VariableTempoBase):
    """Class combining `VariableTempo` objects into a break-point function.

    Class contains methods for converting between time and beats, as well as
    getting the instantaneous tempo at a given time.
    """

    def __init__(self):
        """Create an empty list to be filled via `add_segment`."""
        self.segments = []
        self._length = 0

    def add_segment(self, tempo_function, segment_length=None):
        """Append a new tempo function to the break-point function.

        `tempo_function` is a `VariableTempo` object.

        Each segment of the break-point function is represented by a `Segment`,
        a namedtuple consisting of:

            `tempo_function`
            `start_time` and `end_time`
            `start_beat` and `end_beat`

        Example:
            Create a break-point function consisting of a constant tempo of 120
            bpm for six beats, a linear deceleration from 120 to 60 bpm over
            six beats, and an exponential acceleration from 60 to 120 bpm over
            six beats:

                >>> vtfs = [VariableTempo('constant', start_tempo=120,
                ...                       num_beats=6),
                ...         VariableTempo('linear', start_tempo=120,
                ...                       end_tempo=60, num_beats=6),
                ...         VariableTempo('exponential', start_tempo=60,
                ...                       end_tempo=120, num_beats=6)]
                >>> bpf = VariableTempoBPF()
                >>> for vtf in vtfs:
                ...     bpf.add_segment(vtf)

        """
        if len(self.segments) == 0:
            start_time = 0
            start_beat = 0
        else:
            start_time = self.segments[-1].end_time
            start_beat = self.segments[-1].end_beat

        if segment_length is None:
            segment_length = tempo_function.length
        end_time = start_time + segment_length
        end_beat = tempo_function.time_to_beat(segment_length) + start_beat

        Segment = namedtuple("Segment",
                             "tempo_function start_time end_time "
                             "start_beat end_beat")
        self.segments.append(Segment(tempo_function,
                                     start_time, end_time,
                                     start_beat, end_beat))
        self._length = self.segments[-1].end_time

    @property
    def length(self):
        """length is a read-only property."""
        return self._length

    def time_to_beat(self, t):
        """Convert time to beat."""
        segment = self._get_segment_from_time(t)
        f = segment.tempo_function.time_to_beat
        return f(t - segment.start_time) + segment.start_beat

    def beat_to_time(self, b):
        """Convert beat to time."""
        segment = self._get_segment_from_beat(b)
        f = segment.tempo_function.beat_to_time
        return f(b - segment.start_beat) + segment.start_time

    # helper functions to find the correct segment given a time or beat
    def _get_segment_from_time(self, t):
        for segment in self.segments:
            if segment.start_time <= t <= segment.end_time:
                break
        return segment

    def _get_segment_from_beat(self, b):
        for segment in self.segments:
            if segment.start_beat <= b <= segment.end_beat:
                break
        return segment

    def instantaneous_tempo(self, t):
        """Return instantaneous tempo at time t."""
        segment = self._get_segment_from_time(t)
        return segment.tempo_function.instantaneous_tempo(t -
                                                          segment.start_time)

    def __repr__(self):
        string = "Break-point function with segments:\n\n"
        for i, seg in enumerate(self.segments):
            string += "Segment {} ".format(i + 1)
            string += "start_time={} (start_beat={}), " \
                      .format(round(seg.start_time * 60, 3),
                              seg.start_beat)
            string += "end_time={} (end_beat={})\n" \
                      .format(round(seg.end_time * 60, 3),
                              seg.end_beat)
            string += "{}\n\n".format(seg.tempo_function)
        return string


def transform_stream(original_stream, tempo_function):
    """Transform a music21 stream by a given tempo function.

    Returns a new (flat) music21 stream that is a (deep)copy of the
    `original_stream` with offsets and quarterLength durations modified
    by `tempo_function`.

    `tempo_function` can be either a `VariableTempo` or `VariableTempoBPF`
    object.
    """
    new_stream = m21.stream.Stream()
    # Insert MM = 60 so that offsets can be interpreted as seconds.
    new_stream.append(m21.tempo.MetronomeMark(number=60))
    for e in original_stream.flat:
        new_e = copy.deepcopy(e)
        start = 60 * tempo_function.beat_to_time(e.offset)
        if e.duration is None:
            new_e.duration = None
        else:
            end = 60 * tempo_function.beat_to_time(e.offset +
                                                   e.duration.quarterLength)
            new_e.duration.quarterLength = end - start
        new_stream.insert(start, new_e)
    return new_stream


def main():
    """Examples demonstrating the use of the variable_tempo module."""

    # EXAMPLE 1: Variable tempo function as a mathematical expression.
    # Create expression tempo function.
    expr = "60 * (2 * t - 1) ** 2 + 60"
    vtf = VariableTempo('expression', expr=expr, length=1)
    print(vtf, "\n")

    print("Times for the first 40 beats:")
    for b in range(41):
        t = round(60 * vtf.beat_to_time(b), 3)
        print("Beat {} at {} seconds".format(b, t))

    vtf.graph()
    print("\n\n")

    # EXAMPLE 2: Construct a variable-tempo canon.
    # Create variable tempo function
    tempo_curve = VariableTempo('exponential',
                                start_tempo=60, end_tempo=120, num_beats=6)

    # Create variable tempo break-point function
    # and add three copies of tempo_curve.
    bpf = VariableTempoBPF()
    for _ in range(4):
        bpf.add_segment(tempo_curve)
    # bpf.graph()

    # Parse theme, add tempo marking and show.
    theme = m21.converter.parse('theme.xml')
    theme.insert(0, m21.tempo.MetronomeMark(number=60))
    theme.show()

    # Transform theme by tempo function.
    transformed_theme = transform_stream(theme, bpf)
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
    main()