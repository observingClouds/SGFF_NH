import datetime as dt

import numpy as np


def postprocess_classification_ds(ds, startdate="2018-01-01"):
    """
    Create time coordinate from variable `days` which counts
    the days since a reference date.
    """
    assert np.all(ds.days.values % 1 == 0), "Fractional days do not work currently"
    date = dt.datetime.strptime(startdate, "%Y-%m-%d")
    times = [date + dt.timedelta(days=d.item(0)) for d in ds.days.values.astype(int)]
    ds["time"] = times
    return ds
