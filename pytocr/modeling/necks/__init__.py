__all__ = ["build_neck"]

def build_neck(config):
    from .fpn import FPN
    from .fpem_ffm import FPEM_FFM
    from .rnn import SequenceEncoder
    from .csp_pan import CSPPAN
    support_dict = ["FPN", "FPEM_FFM", "SequenceEncoder", "CSPPAN"]

    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "neck only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
