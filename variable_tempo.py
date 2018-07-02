"""Classes for working with continuous variable tempo functions."""

import music21 as m21
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import math
from collections import namedtuple
import copy


class VariableTempo(ABC):
    """Abstract base class for handling continuously variable tempos.

    Class contains abstract methods for converting between time and beats, as
    well as methods for getting the instantaneous tempo at a given time and
    graphing the tempo function.

    Attributes:
        start_tempo (float): Initial tempo for the variable tempo function.
            Defaults to `None`. `start_tempo` is assumed to be in beats per
            minute and is required for `ConstantVariableTempo`,
            `LinearVariableTempo`, and `ExponentialVariableTempo` subclasses.

        end_tempo (float): Ending tempo for the variable tempo function.
            Defaults to `None`. `end_tempo` is assumed to be in beats per
            minute and is a required attribute for `LinearVariableTempo` and
            `ExponentialVariableTempo` subclasses.

        length (float): Length of time (in minutes) from `start_tempo` to
            `end_tempo`. Defaults to `None`. `LinearVariableTempo` and
            `ExponentialVariableTempo` subclasses require a value for either
            `length` or `num_beats` (see below). If a value for `length` is
            not provided, it is derived from `num_beats`.

        num_beats (float): Number of beats occuring between `start_tempo` and
            `end_tempo`. Defaults to `None`. `LinearVariableTempo` and
            `ExponentialVariableTempo` subclasses require a value for either
            `num_beats` or `length` (see above). If a value for `num_beats` is
            not provided, it is derived from `length`.

        expr (str): `expr` must be a string that gives rise to a valid
            mathematical expression of the single variable 't' when evaluated
            by the `eval` function. `expr` is required by the
            `ExpressionVariableTempo` subclass. For the other subclasses,
            `expr` is derived.

    """

    def __init__(self, start_tempo=None, end_tempo=None,
                 length=None, num_beats=None, expr=None):
        self.start_tempo = start_tempo
        self.end_tempo = end_tempo
        self.length = length
        self.num_beats = num_beats
        self.expr = expr

    @abstractmethod
    def time_to_beat(self, t):
        """Given a time in minutes, returns the corresponding beat.

        The beat is the definite integral of the tempo function from 0 to `t`.

        Example:
            >>> vtf = LinearVariableTempo(start_tempo=60, end_tempo=120,
            ...                           length=1)
            >>> vtf.time_to_beat(0.5)
            37.5

        """
        pass

    @abstractmethod
    def beat_to_time(self, b):
        """Given a beat, returns the corresponding time in minutes.

        Inverse function of .time_to_beat. The corresponding time is determined
        by setting the definite integral of f(t) from 0 to t_0 = b and solving
        for t_0.

        Example:
            >>> vtf = LinearVariableTempo(start_tempo=60, end_tempo=120,
            ...                           length=1)
            >>> vtf.beat_time_tempo(37.5)
            0.5

        """
        pass

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

    def instantaneous_tempo(self, t):
        """Return instantaneous tempo at time t."""
        return eval(self.expr)

    def __repr__(self):
        arg_names = ["start_tempo", "end_tempo",
                     "length", "num_beats", "expr"]
        name = self.__class__.__name__
        attr = ", ".join("{}={!r}".format(k, v)
                         for k, v in self.__dict__.items()
                         if k in arg_names and v is not None)
        return "{}({})".format(name, attr)

    def __str__(self):
        return "{}: f(t) = {}".format(self.__class__.__name__, self.expr)


class ConstantVariableTempo(VariableTempo):
    """Subclass for constant variable tempo functions.

    Required attributes:
        `start_tempo`

    Example:
        Create a `VariableTempo` object for a constant tempo of 60 beats per
        minute--f(t) = 60:

            >>> curve = ConstantVariableTempo(start_tempo=60)

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set expr.
        self.expr = "{}".format(self.start_tempo)

        # Set length if num_beats is not None.
        if self.num_beats:
            self.length = self.num_beats / self.start_tempo

    def time_to_beat(self, t):
        return self.start_tempo * t

    def beat_to_time(self, beat):
        return beat / self.start_tempo

    def __str__(self):
        prefix = super().__str__()
        return "{}\nTempo is {}bpm.".format(prefix, self.start_tempo)


# Shared methods for LinearVariableTempo and ExponentialVariableTempo classes
class _LinearAndExponentialVariableTempos(VariableTempo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Check that 'linear' and 'exponential' functions have a value other
        # than None for either length or num_beats
        if not (self.length or self.num_beats):
            raise ValueError("LinearVariableTempo and ExponentialVariableTempo"
                             "classes require a value for either length or "
                             "num_beats.")

    def __str__(self):
        prefix = super().__str__()
        description = ("\nTempo is {}bpm to {}bpm over {} seconds ({} beats)."
                       .format(self.start_tempo, self.end_tempo,
                               round(self.length * 60, 3),
                               round(self.num_beats, 3)))
        return prefix + description


class LinearVariableTempo(_LinearAndExponentialVariableTempos):
    """Subclass for linear variable tempo functions.

    Required attributes:
        `start_tempo`
        `end_tempo`
        `length` or `num_beats`

    Example:
        Create a `VariableTempo` object for a linear acceleration from 60 bpm
        to 120 bpm over 0.5 minutes--f(t) = 60 * 2 * t + 60:

            >>> curve = LinearVariableTempo(start_tempo=60, end_tempo=120,
            ...                             length=0.5)

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def time_to_beat(self, t):
        return self.slope * t**2 / 2 + self.start_tempo * t

    def beat_to_time(self, beat):
        a = self.slope / 2
        b = self.start_tempo
        c = -beat
        return (-b + (b**2 - 4 * a * c)**0.5) / (2 * a)


class ExponentialVariableTempo(_LinearAndExponentialVariableTempos):
    """Subclass for exponential variable tempo functions.

    Required attributes:
        `start_tempo`
        `end_tempo`
        `length` or `num_beats`

    Example:
        Create an `VariableTempo` object for an exponential acceleration from
        60 bpm to 120 bpm over 2 minutes--f(t) = 60 + (120/60)^(t/2):

            >>> curve = ExponentialVariableTempo(start_tempo=60, end_tempo=120,
            ...                                  length=2)

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def time_to_beat(self, t):
        a = self.start_tempo
        r = self.ratio
        L = self.length
        return (a * L / math.log(r)) * (r**(t / L) - 1)

    def beat_to_time(self, beat):
        # Equation can yield ``ValueError: math domain error`` for beats
        # below a certain value. Need to determine exact value given a, r, L.
        a = self.start_tempo
        r = self.ratio
        L = self.length
        b = beat
        return L * math.log((b * math.log(r)) / (a * L) + 1, r)


class ExpressionVariableTempo(VariableTempo):
    """Subclass for variable tempo functions based on a given expression.

    Required attributes:
        `expr`

    Subclass depends on the `quad` function in `scipy.integrate`.

    Example:
        Create a `VariableTempo` object for a tempo of the form
        f(t) = 60t^2 + 60:

            >>> curve = ExpressionVTF(expr='60 * t**2 + 60')

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set expression if expr is valid.
        if self.expr is not None:
            # test if expr can be evaluated by eval
            try:
                t = 1
                eval(self.expr)
            except:
                raise ValueError("Invalid expression for expr")
        else:
            raise ValueError("self.expr cannot be None for {}."
                             .format(self.__class__.__name__))

    def time_to_beat(self, t):
        """Return the result of integration with `scipy.integrate.quad`."""
        f = lambda t: eval(self.expr)
        return scipy.integrate.quad(f, 0, t)[0]

    def beat_to_time(self, beat):
        """Binary search using the inverse method."""
        max_error = 0.0001  # Fix. max_error should correspond to nearest ms.
        min_time = 0
        max_time = 0
        max_beat = 0

        while max_beat < beat:
            min_time = max_time
            max_time += 1
            max_beat = self.time_to_beat(max_time)
        beat_error = 1
        while beat_error > max_error:
            mid_time = (min_time + max_time) / 2.0
            mid_beat = self.time_to_beat(mid_time)
            if mid_beat < beat:
                min_time = mid_time
            else:
                max_time = mid_time
            beat_error = abs(mid_beat - beat)
        return mid_time


class NancarrowGeometricAcceleration(VariableTempo):
    """VariableTempo class for Nancarrow's geometric accleration.

    Nancarrow's geometric acceleration is defined by an initial duration and
    a consistent ratio between consecutive durations. Given an initial duration
    (d) and ratio (r), the sequence of durations is:
        d, d * r, d * r**2, d * r**3, ...

    For example, given an initial duration of 2 seconds (d=2) and an
    acceleration by 5% (r=1/1.05), the sequence of durations is:
        2, 1.905, 1.814, 1.728, ...

    There are a number of problems with modeling acceleration as a recursive
    operation on duration rather than a continuous function of tempo. (See
    Callender. 2001. “Formalized Accelerando: An Extension of Rhythmic
    Techniques in Nancarrow’s Acceleration Canons.” Perspectives of New Music
    39, no. 1: 188-210.)

    `NancarrowGeometricAcceleration` is a subclass of
    `ExpressionVariableTempo`. The parameters are used by the constructor
    below to recast geometric acceleration as a continuous tempo function:

                f(t) = -\frac{1 - r}{\ln(r)(d - t(1 - r))}.

    Parameters:
        initial_duration (float)
        ratio (float)

    """
    def __init__(self, initial_duration, percent):
        self.percent = percent
        self.d = initial_duration / 60

        # Derive ratio from percent. For example, by Nancarrow's reckoning
        # a 5% acceleration is the ratio 1 / 1.05;
        # a -5% acceleration is the ratio 1.05.
        if percent > 0:
            self.r = 1 / (1 + percent / 100)
        else:
            self.r = 1 + percent / 100

        # f(t) = -\frac{1 - r}{\ln(r)(d - t(1 - r))}
        expr = ("-(1 - {1}) / (math.log({1}) * ({0} - t * (1 - {1})))"
                .format(self.d, self.r))
        super().__init__(expr=expr)

    def time_to_beat(self, t):
        # \int f(t) = \log_r (1-\frac{t(1-r)}{d})
        return math.log(1 - (t * (1 - self.r) / self.d), self.r)

    def beat_to_time(self, b):
        # t = \frac{d(1 - r^b)}{1 - r}
        return self.d * (1 - self.r**b) / (1 - self.r)

    def __str__(self):
        prefix = super().__str__()
        description = ("\nInitial duration = {} seconds, acceleration = {}%."
                       .format(self.d * 60, self.percent))
        return prefix + description


class VariableTempoBPF(VariableTempo):
    """Class combining `VariableTempo` objects into a break-point function.

    Class contains methods for converting between time and beats, as well as
    getting the instantaneous tempo at a given time. Inherits the `graph`
    method from the `VariableTempo` class.

    """

    def __init__(self, segments=[]):
        """Create an empty list to be filled via `add_segment`."""
        self.segments = []
        self._length = 0

        # Add segments to self.segments
        for segment in segments:
            self.append(segment)

    def append(self, tempo_function, segment_length=None):
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

                >>> vtfs = [ConstantVariableTempo(start_tempo=120,
                ...                               num_beats=6),
                ...         LinearVariableTempo(start_tempo=120,
                ...                             end_tempo=60, num_beats=6),
                ...         ExponentialVariableTempo(start_tempo=60,
                ...                                  end_tempo=120,
                ...                                  num_beats=6)]
                >>> bpf = VariableTempoBPF()
                >>> for vtf in vtfs:
                ...     bpf.append(vtf)

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
        """`length` is a read-only property."""
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
        name = self.__class__.__name__
        segments = ", ".join(segment.tempo_function.__repr__()
                             for segment in self.segments)
        return f"{name}([{segments}])"

    def __str__(self):
        prefix = "Break-point function with segments\n"
        description = ("".join(f"Segment {i + 1}:\n{seg}\n\n"
                               for i, seg in enumerate(self.segments)))
        return prefix + description


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
    vtf = ExpressionVariableTempo(expr=expr, length=1)
    print(vtf, "\n")

    print("Times for the first 40 beats:")
    for b in range(41):
        t = round(60 * vtf.beat_to_time(b), 3)
        print("Beat {} at {} seconds".format(b, t))

    vtf.graph()
    print("\n\n")

    # EXAMPLE 2: Construct a variable-tempo canon.
    # Create variable tempo function
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
    # theme.show()

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

    print("\n\n")

    # EXAMPLE 3: Nancarrow geometric acceleration
    nancarrow = NancarrowGeometricAcceleration(initial_duration=2, percent=5)
    print(nancarrow)
    for beat in range(7):
        print(nancarrow.beat_to_time(beat) * 60)  # 0, 2, 3.905, 5.719, ...


if __name__ == "__main__":
    main()
