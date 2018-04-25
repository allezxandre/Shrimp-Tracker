import matplotlib.pyplot as plt
import numpy as np


class Tracer(object):

    def __init__(self, row_names=None):
        self.states = None
        self._state_dim = None
        self._current_row = None
        self.row_names = row_names

    def _prepare_states(self, state_dim: int, forIndex: int):
        if self.states is None or self._state_dim is None:
            self._state_dim = state_dim
            self._current_row = np.zeros((1), dtype=int)
            self.states = np.zeros((1, self._state_dim))
        assert self._state_dim == state_dim
        _, nb_cols = self._column_indices(forIndex)
        nb_rows, old_nb_cols = self.states.shape
        if nb_cols > old_nb_cols:
            # The index is not yet in the states array, add columns
            new_states_array = np.zeros((nb_rows, nb_cols))
            new_states_array[:, :old_nb_cols] = self.states
            self.states = new_states_array
            new_current_row = np.zeros((nb_cols), dtype=int)
            # TODO: Use (nb_rows * np.ones) to have each new track start on the last row
            new_current_row[:old_nb_cols] = self._current_row
            self._current_row = new_current_row
        assert self.states.shape[1] == len(self._current_row)

    def _column_indices(self, forIndex: int):
        begin = forIndex * self._state_dim
        end = begin + self._state_dim
        return begin, end

    def register(self, state: np.ndarray, forIndex: int):
        assert state.ndim == 1, "State is supposed to be 1-dimensionnal"
        state = state.flatten()
        state_dim = state.shape[0]
        self._prepare_states(state_dim, forIndex)
        self.add(state, forIndex)

    def add(self, state, forIndex: int):
        assert state.ndim == 1, "State is supposed to be 1-dimensionnal"
        state = state.flatten()
        assert state.shape[0] == self._state_dim, "State has an unexpected dimension"
        row = self._current_row[forIndex]
        if row >= self.states.shape[0]:
            # We need to add row(s) first:
            nb_missing_rows = row - self.states.shape[0] + 1
            new_rows = np.zeros((nb_missing_rows, self.states.shape[1]))
            self.states = np.append(self.states, new_rows, axis=0)
        # We can write in the states array
        begin, end = self._column_indices(forIndex)
        self.states[row, begin:end] = state
        self._current_row[forIndex] += 1

    def track(self, forIndex):
        begin, end = self._column_indices(forIndex)
        nb_rows = self._current_row[forIndex]
        return self.states[:nb_rows, begin:end]

    def tracks(self):
        return [self.track(forIndex) for forIndex in range(len(self._current_row))]

    def write(self, minimum_length=10):
        pass


class TracerCSV(Tracer):
    def __init__(self, row_names=None, output_CSV_path=None):
        super().__init__(row_names)
        self.output_CSV_path = output_CSV_path

    def write(self, minimum_length=10):
        if self.output_CSV_path is None: return
        dim = self._state_dim
        states_reworked = np.empty(self.states.shape, dtype=self.states.dtype).reshape((-1, dim))
        last_row = 0
        for trackIndex, nb_rows in enumerate(self._current_row):
            begin, end = self._column_indices(trackIndex)
            nb_rows = self._current_row[trackIndex]
            if nb_rows > 0:
                states_reworked[last_row:last_row+nb_rows,:] = self.states[:nb_rows, begin:end]
            last_row += nb_rows
        header=None
        if self.row_names is not None:
            header = ''.join(self.row_names)
        np.savetxt(self.output_CSV_path, states_reworked, delimiter=",", header=header)


class TracerPlot(Tracer):
    def write(self, minimum_length=10):
        tracks = self.tracks()
        tracks = [track for track in tracks if len(track) > minimum_length]
        for track in tracks:
            mean = [track.mean(axis=0) for track in tracks]
            std_dev = [track.std(axis=0) for track in tracks]
        # Plot speed
        for track in tracks:
            plt.plot(np.abs(track[:, 6]))
        plt.xlim([0, 100])
        plt.ylim([0, 75])
        plt.yticks(np.arange(0, 75, 5))
        plt.grid()
        plt.show()
        # Plot angular velocity
        for track in tracks:
            plt.plot(np.abs(track[:, 7]))
        plt.xlim([0, 15])
        ymax = 0.3
        plt.ylim([0, ymax])
        plt.yticks(np.arange(0, ymax, 0.025))
        plt.grid()
        plt.show()
        # Plot angle
        for track in tracks:
            plt.plot(np.abs(track[:, 2]))
        plt.xlim([0, 15])
        ymax = np.pi / 2
        plt.ylim([0, ymax])
        plt.yticks(np.arange(0, ymax, np.pi / 24))
        plt.grid()
        plt.show()
        # Plot angle difference
        for track in tracks:
            plt.plot(np.abs(track[2::2, 2] - track[:-2:2, 2]))
        plt.xlim([0, 15])
        ymax = np.pi / 2
        plt.ylim([0, ymax])
        plt.yticks(np.arange(0, ymax, np.pi / 24))
        plt.grid()
        plt.show()
        # Plot area
        for track in tracks:
            plt.plot(np.abs(track[2::2, 3] - track[:-2:2, 3]))
        plt.xlim([0, 50])
        ymin, ymax = 0, 50
        plt.ylim([ymin, ymax])
        plt.yticks(np.arange(ymin, ymax, 5))
        plt.grid()
        plt.show()
        # Plot lambdas
        for track in tracks:
            plt.plot(np.abs(track[2::2, 4] - track[:-2:2, 4]))
        plt.xlim([0, 50])
        ymin, ymax = 0, 50
        plt.ylim([ymin, ymax])
        plt.yticks(np.arange(ymin, ymax, 5))
        plt.grid()
        plt.show()
