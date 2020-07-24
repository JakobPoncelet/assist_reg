'''@file acoustic_factory.py
contains the acoustic model factory'''

def factory(name):
    '''the acousic modef factory method

    args:
        name: the name of the class

    returns:
        an acoustic model class
    '''

    if name == 'gmm':
        from . import gmm
        return gmm.GMM
    else:
        raise Exception('unknown acoustic model: %s' % name)
