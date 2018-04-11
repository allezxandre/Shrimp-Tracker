def generate_color_set(n):
    from numpy import linspace
    import matplotlib.cm as cm
    return [tuple(map(int, c)) for c in (cm.gist_rainbow(linspace(0, 1, n)) * 255)[:, :3]]


def side_by_side(left, right, axis=1, separator_line_width=3, separator_intensity=255 // 2):
    from numpy import concatenate, full
    other = 0 if axis == 1 else 1
    assert left.shape[other] == right.shape[other], \
        'The left and right should have the same height. ' + \
        'They are respectively {} and {}.'.format(left.shape[other], right.shape[other])
    separator_shape = tuple((separator_line_width if i == axis else l for i, l in enumerate(left.shape)))
    return concatenate((left, full(separator_shape, separator_intensity, dtype=left.dtype), right), axis=1)
