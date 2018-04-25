import pandas as pd


class Tracer(object):

    def __init__(self):
        self.trace: pd.DataFrame = None

    def add(self, trace: pd.DataFrame):
        if self.trace is None:
            self.trace = trace
        else:
            self.trace = self.trace.append(trace)

    def write(self, minimum_length=10):
        pass


class TracerCSV(Tracer):
    def __init__(self, output_CSV_path=None):
        super().__init__()
        self.output_CSV_path = output_CSV_path

    def write(self, minimum_length=10):
        if self.output_CSV_path is None: return
        trace_sorted = self.trace.sort_values(by=['Track ID', 'Timestep'])
        trace_sorted.to_csv(self.output_CSV_path)
