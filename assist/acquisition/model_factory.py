'''@file model_factory.py
contains the model factory'''

def factory(name):
    '''model factory method

    args:
        name: tye type of model as a string

    Returns:
        a model class'''

    if name == 'rccn':
        import tfmodel.rccn
        return tfmodel.rccn.RCCN
    elif name == 'rccn_spk':
        import tfmodel.rccn_spk
        return tfmodel.rccn_spk.RCCN_SPK
    elif name == 'pccn':
        import tfmodel.pccn
        return tfmodel.pccn.PCCN
    elif name == 'pccn_spk':
        import tfmodel.pccn_spk
        return tfmodel.pccn_spk.PCCN_SPK
    elif name == 'encoderdecoder':
        import tfmodel.encoderdecoder
        return tfmodel.encoderdecoder.EncoderDecoder
    elif name == 'nmf':
        import nmf
        return nmf.NMF
    else:
        raise Exception('unknown acquisition type %s' % name)
