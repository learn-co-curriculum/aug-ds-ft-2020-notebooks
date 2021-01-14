import pandas as pd

def print_xy(X,y):
    """Example function to print 2 things

    Args:
        X (list): First thing printed
        y (list): Second thing printed
    Ex:
    >> X = []..
    >> print(X,y)
    """
    print(X)
    print(y)
    
    
def print_xy2(X,y):
    print(X)
    print(y)
    print("not really 2")
    
    
def series_xy_file(X,y):
    return pd.Series(y,index=X)
