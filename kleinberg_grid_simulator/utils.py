from dataclasses import dataclass


@dataclass
class Result:
    """
    Dataclass to represent the results.
    """
    edt: float
    process_time: float
    n: int
    r: float
    p: int
    q: int
    n_runs: int
    julia: bool
    numba: bool = True
    parallel: bool = False

    def __repr__(self):
        exclude = {'numba', 'parallel'} if self.julia else {'julia'}
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items() if key not in exclude]
        return "{}({})".format(type(self).__name__, ", ".join(kws))
