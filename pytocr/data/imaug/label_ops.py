import numpy as np
import json

from pytocr.utils.logging import get_logger

class ClsLabelEncode(object):
    def __init__(self, label_list, **kwargs):
        self.label_list = label_list

    def __call__(self, data):
        label = data["label"]
        if label not in self.label_list:
            return None
        label = self.label_list.index(label)
        data["label"] = label
        return data


class DetLabelEncode(object):
    def __init__(self, ignore_txt=["*", "###"], **kwargs):
        self.ignore_txt = ignore_txt

    def __call__(self, data):
        label = data["label"]
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]["points"]
            # if not ordered, do order_points_clockwise
            # if len(box) == 4:
            #     box = self.order_points_clockwise(
            #         np.array(box, dtype=np.float32))
            txt = label[bno]["transcription"]
            boxes.append(box)
            txts.append(txt)
            if txt in self.ignore_txt:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        boxes = self.expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool)

        data["polys"] = boxes
        data["texts"] = txts
        data["ignore_tags"] = txt_tags
        return data
    
    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def expand_points_num(self, boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes


class BaseRecLabelEncode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 cn2en=False):

        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.lower = False
        self.cn2en = cn2en

        if character_dict_path is None:
            logger = get_logger()
            logger.warning(
                "The character_dict_path is None, model can only recognize number and lower letters"
            )
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
            self.lower = True
        else:
            self.character_str = ""
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str += line
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        if self.lower:
            text = text.lower()
        # 将中英文相似字符统一转换为英文
        if self.cn2en:
            text = text.replace("（","(").replace("）",")").replace("：",":").replace("；",";").replace("！","!").replace("？","?")
        text_list = []
        for char in text:
            if char not in self.dict:
                logger = get_logger()
                logger.warning("{} is not in dict".format(char))
                continue  # 直接删除是否合适？ 可否更换为特定字符(比如"*"")
                # char = "*"
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 cn2en=False,
                 **kwargs):
        super(CTCLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char, cn2en)

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        data["length"] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text)) # 保证长度一样，可以进行batch训练
        data["label"] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1   # 字符分布直方图
        data["label_ace"] = np.array(label)
        return data

    def add_special_char(self, dict_character):
        # 把索引0作为占位符 index 0 is blank  blank_idx=0
        dict_character = ["blank"] + dict_character
        return dict_character


# ----------------------- NRTR -------------------------#
# class NRTRLabelEncode(BaseRecLabelEncode):
#     """ Convert between text-label and text-index """

#     def __init__(self,
#                  max_text_length,
#                  character_dict_path=None,
#                  use_space_char=False,
#                  **kwargs):

#         super(NRTRLabelEncode, self).__init__(
#             max_text_length, character_dict_path, use_space_char)

#     def __call__(self, data):
#         text = data['label']
#         text = self.encode(text)
#         if text is None:
#             return None
#         if len(text) >= self.max_text_len - 1:
#             return None
#         data['length'] = np.array(len(text))
#         text.insert(0, 2)
#         text.append(3)
#         text = text + [0] * (self.max_text_len - len(text))
#         data['label'] = np.array(text)
#         return data

#     def add_special_char(self, dict_character):
#         dict_character = ['blank', '<unk>', '<s>', '</s>'] + dict_character
#         return dict_character


# ----------------------- SRN -------------------------#
# class SRNLabelEncode(BaseRecLabelEncode):
#     """ Convert between text-label and text-index """

#     def __init__(self,
#                  max_text_length=25,
#                  character_dict_path=None,
#                  use_space_char=False,
#                  **kwargs):
#         super(SRNLabelEncode, self).__init__(
#             max_text_length, character_dict_path, use_space_char)

#     def add_special_char(self, dict_character):
#         dict_character = dict_character + [self.beg_str, self.end_str]
#         return dict_character

#     def __call__(self, data):
#         text = data['label']
#         text = self.encode(text)
#         char_num = len(self.character)
#         if text is None:
#             return None
#         if len(text) > self.max_text_len:
#             return None
#         data['length'] = np.array(len(text))
#         text = text + [char_num - 1] * (self.max_text_len - len(text))
#         data['label'] = np.array(text)
#         return data

#     def get_ignored_tokens(self):
#         beg_idx = self.get_beg_end_flag_idx("beg")
#         end_idx = self.get_beg_end_flag_idx("end")
#         return [beg_idx, end_idx]

#     def get_beg_end_flag_idx(self, beg_or_end):
#         if beg_or_end == "beg":
#             idx = np.array(self.dict[self.beg_str])
#         elif beg_or_end == "end":
#             idx = np.array(self.dict[self.end_str])
#         else:
#             assert False, "Unsupport type %s in get_beg_end_flag_idx" \
#                           % beg_or_end
#         return idx


# ----------------------- SAR -------------------------#
# class SARLabelEncode(BaseRecLabelEncode):
#     """ Convert between text-label and text-index """

#     def __init__(self,
#                  max_text_length,
#                  character_dict_path=None,
#                  use_space_char=False,
#                  **kwargs):
#         super(SARLabelEncode, self).__init__(
#             max_text_length, character_dict_path, use_space_char)

#     def add_special_char(self, dict_character):
#         beg_end_str = "<BOS/EOS>"
#         unknown_str = "<UKN>"
#         padding_str = "<PAD>"
#         dict_character = dict_character + [unknown_str]
#         self.unknown_idx = len(dict_character) - 1
#         dict_character = dict_character + [beg_end_str]
#         self.start_idx = len(dict_character) - 1
#         self.end_idx = len(dict_character) - 1
#         dict_character = dict_character + [padding_str]
#         self.padding_idx = len(dict_character) - 1

#         return dict_character

#     def __call__(self, data):
#         text = data['label']
#         text = self.encode(text)
#         if text is None:
#             return None
#         if len(text) >= self.max_text_len - 1:
#             return None
#         data['length'] = np.array(len(text))
#         target = [self.start_idx] + text + [self.end_idx]
#         padded_text = [self.padding_idx for _ in range(self.max_text_len)]

#         padded_text[:len(target)] = target
#         data['label'] = np.array(padded_text)
#         return data

#     def get_ignored_tokens(self):
#         return [self.padding_idx]
