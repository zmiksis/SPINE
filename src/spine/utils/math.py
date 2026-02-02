# Math utility functions

def integrate(f, f0, dx, f_total=None):

    if f_total is None:
        f_total = (dx / 2) * (f + f0)
    else:
        f_total += (dx / 2) * (f + f0)
    return f_total