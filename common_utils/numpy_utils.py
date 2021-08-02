import json
import numpy as np

# Source: https://stackoverflow.com/a/47626762
class NumpyEncoder(json.JSONEncoder):
    """ This class allows proper serialization from Numpy arrays to JSON. """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def numpy_jsonify(arr):
    """ Deeply converts an object with numpy variables into their respective native python types for JSON encoding. """
    return json.loads(json.dumps(arr, cls=NumpyEncoder))


# Describes basic statistics of a data array
def stats_describe(arr):
    uniq, counts = np.unique(arr, return_counts=True)
    counts_dict = dict(zip(numpy_jsonify(uniq), numpy_jsonify(counts)))
    stats = {
        'count': len(arr),
        'counts': counts_dict,
        'min': np.min(arr).item(),
        'max': np.max(arr).item(),
        'median': np.median(arr).item(),
        'mean': np.mean(arr).item(),
        'stdev': np.std(arr).item(),
    }

    return stats