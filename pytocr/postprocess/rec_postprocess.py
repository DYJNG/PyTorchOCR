import numpy as np
import torch
import torch.nn.functional as F

class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"

        self.character_str = []
        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("UTF-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(
        self, 
        character_dict_path=None, 
        use_space_char=False, 
        **kwargs
    ):
        super(CTCLabelDecode, self).__init__(
            character_dict_path, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple):
            preds = preds[-1]
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()  # T(len_seq) * batch_size * num_class
            preds = preds.transpose((1, 0, 2))    # 转换为 batch_size * T * num_class 
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)  # np.max()只返回值 torch.max()返回索引和值
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:   # when infer
            return text
        label = self.decode(label)   # idx -> char  when train
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


class DistillationCTCLabelDecode(CTCLabelDecode):
    """
    Convert 
    Convert between text-label and text-index
    """

    def __init__(
        self, 
        character_dict_path=None, 
        use_space_char=False, 
        model_name=["student"], 
        key=None, 
        **kwargs
    ):
        super(DistillationCTCLabelDecode, self).__init__(
            character_dict_path, use_space_char)
        if not isinstance(model_name, list):
            model_name = [model_name]
        self.model_name = model_name

        self.key = key

    def __call__(self, preds, label=None, *args, **kwargs):
        output = dict()
        for name in self.model_name:
            pred = preds[name]
            if self.key is not None:
                pred = pred[self.key]
            output[name] = super().__call__(pred, label=label, *args, **kwargs)
        return output


class AttnLabelDecode(BaseRecLabelDecode):
    def __init__(
        self, 
        character_dict_path=None, 
        use_space_char=False, 
        **kwargs
    ):
        super(AttnLabelDecode, self).__init__(
            character_dict_path, use_space_char)
    
    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character
    
    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        end_idx = ignored_tokens[1]
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx
        