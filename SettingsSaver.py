import pickle
from os import path
from pathlib import Path

class SettingsSaver(object):

    def __init__(self):
        from appdirs import AppDirs
        app_dirs = AppDirs('ShrimpTracker', 'GeorgiaTech')
        self.cache_dir = app_dirs.user_cache_dir
        self.data_dir = app_dirs.user_data_dir
        self.cache = {}
        self.__load_cache()


    def __assert_exists(self):
        for directory in [self.cache_dir, self.data_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @property
    def cache_file(self):
        return path.join(self.cache_dir, 'cache.pkl')

    def __load_cache(self):
        try:
            with open(self.cache_file, 'rb') as f:
                cache_object = pickle.load(f)
                if type(cache_object) is dict:
                    self.cache = cache_object
                else:
                    self.cache = {}
        except FileNotFoundError:
            print('Cache does not exist')
            self.cache = {}

    def add_to_cache(self, filename, kalman, circle):
        self.cache[filename] = (kalman, None)
        if circle is not None:
            self.cache[filename] = (kalman, circle)
        self.save_cache()

    def read_from_cache(self, filename):
        if filename in self.cache:
            return self.cache[filename]
        else:
            return None

    def save_cache(self):
        self.__assert_exists()
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f, pickle.HIGHEST_PROTOCOL)