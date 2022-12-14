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
                 lower=False, 
                 cn2en=False):

        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.lower = lower
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
                    line = line.decode("UTF-8").strip("\n").strip("\r\n")
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
        # ?????????????????????????????????????????????
        if self.cn2en:
            text = text.replace("???","(").replace("???",")").replace("???",":").replace("???",";").replace("???","!").replace("???","?")
        text_list = []
        for char in text:
            if char not in self.dict:
                logger = get_logger()
                logger.warning("{} is not in dict".format(char))
                continue  # ??????????????????????????? ???????????????????????????(??????"*"")
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
        text = text + [0] * (self.max_text_len - len(text)) # ?????????????????????????????????batch??????
        data["label"] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1   # ?????????????????????
        data["label_ace"] = np.array(label)
        return data

    def add_special_char(self, dict_character):
        # ?????????0??????????????? index 0 is blank  blank_idx=0
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


class AttnLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """
    def __init__(
        self, 
        max_text_length, 
        character_dict_path=None, 
        use_space_char=False, 
        **kwargs
    ):
        super(AttnLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)
    
    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character
    
    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len:
            return None
        data["length"] = np.array(len(text))
        text = [0] + text + [len(self.character) - 1] + [0] * (
            self.max_text_len - len(text) - 2)  # sos + text + eos + padding
        data["label"] = np.array(text)
        return data
    

class TableLabelEncode(AttnLabelEncode):
    """ Convert between text-label and text-index """
    
    def __init__(
        self, 
        max_text_length, 
        character_dict_path, 
        replace_empty_cell_token=False, 
        merge_no_span_structure=False,
        learn_empty_box=False, 
        loc_reg_num=4, 
        **kwargs
    ):
        super(TableLabelEncode, self).__init__(
            max_text_length, character_dict_path)
        self.max_text_len = max_text_length
        # self.lower = False
        # self.cn2en = False
        self.learn_empty_box = learn_empty_box
        self.merge_no_span_structure = merge_no_span_structure
        self.replace_empty_cell_token = replace_empty_cell_token

        dict_character = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode("UTF-8").strip("\n").strip("\r\n")
                dict_character.append(line)
        
        if self.merge_no_span_structure:
            if "<td></td>" not in dict_character:
                dict_character.append("<td></td>")
            if "<td>" in dict_character:
                dict_character.remove("<td>")
        
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.idx2char = {v: k for k, v in self.dict.items()}

        self.character = dict_character
        self.loc_reg_num = loc_reg_num
        self.pad_idx = self.dict[self.beg_str]
        self.start_idx = self.dict[self.beg_str]
        self.end_idx = self.dict[self.end_str]

        self.td_token = ["<td>", "<td", "<eb></eb>", "<td></td>"]
        self.empty_bbox_token_dict = {
            "[]": "<eb></eb>", 
            "[' ']": "<eb1></eb1>", 
            "['<b>', ' ', '</b>']": "<eb2></eb2>", 
            "['\\u2028', '\\u2028']": "<eb3></eb3>",
            "['<sup>', ' ', '</sup>']": "<eb4></eb4>",
            "['<b>', '</b>']": "<eb5></eb5>",
            "['<i>', ' ', '</i>']": "<eb6></eb6>",
            "['<b>', '<i>', '</i>', '</b>']": "<eb7></eb7>",
            "['<b>', '<i>', ' ', '</i>', '</b>']": "<eb8></eb8>",
            "['<i>', '</i>']": "<eb9></eb9>",
            "['<b>', ' ', '\\u2028', ' ', '\\u2028', ' ', '</b>']":
            "<eb10></eb10>"
        }

    @property
    def _max_text_len(self):
        return self.max_text_len + 2

    def __call__(self, data):
        cells = data["cells"]
        structure = data["structure"]
        if self.merge_no_span_structure:
            structure = self._merge_no_span_structure(structure)
        if self.replace_empty_cell_token:
            structure = self._replace_empty_cell_token(structure, cells)
        # remove empty token and add " " to span token
        new_structure = []
        for token in structure:
            if token != "":
                if "span" in token and token[0] != " ":
                    token = " " + token
                new_structure.append(token)
        # encode structure
        structure = self.encode(new_structure)
        if structure is None:
            return None
        structure = [self.start_idx] + structure + [self.end_idx]
        structure = structure + [self.pad_idx] * (
            self._max_text_len - len(structure))  # pad
        structure = np.array(structure)
        data["structure"] = structure
        if len(structure) > self._max_text_len:
            return None
        
        # encode box
        bboxes = np.zeros(
            (self._max_text_len, self.loc_reg_num), dtype=np.float32)
        bbox_masks = np.zeros(
            (self._max_text_len, 1), dtype=np.float32)
        bbox_idx = 0
        for i, token in enumerate(structure):
            if self.idx2char[token] in self.td_token:
                if "bbox" in cells[bbox_idx] and len(
                    cells[bbox_idx]["tokens"]) > 0:
                    bbox = cells[bbox_idx]["bbox"].copy()
                    bbox = np.array(bbox, dtype=np.float32).reshape(-1)
                    bboxes[i] = bbox
                    bbox_masks[i] = 1.0
                if self.learn_empty_box:
                    bbox_masks[i] = 1.0
                bbox_idx += 1
        data["bboxes"] = bboxes
        data["bbox_masks"] = bbox_masks
        return data
    
    def _merge_no_span_structure(self, structure):
        new_structure = []
        i = 0
        while i < len(structure):
            token = structure[i]
            if token == "<td>":
                token = "<td></td>"
                i += 1
            new_structure.append(token)
            i += 1
        return new_structure
    
    def _replace_empty_cell_token(self, token_list, cells):
        bbox_idx = 0
        add_empty_bbox_token_list = []
        for token in token_list:
            if token in ["<td></td>", "<td", "<td>"]:
                if "bbox" not in cells[bbox_idx].keys():
                    content = str(cells[bbox_idx]["tokens"])
                    token = self.empty_bbox_token_dict[content]
                add_empty_bbox_token_list.append(token)
                bbox_idx += 1
            else:
                add_empty_bbox_token_list.append(token)
        return add_empty_bbox_token_list


class TableBoxEncode(object):
    def __init__(
        self, 
        in_box_format="xyxy", 
        out_box_format="xyxy", 
        **kwargs
    ):
        assert out_box_format in ["xywh", "xyxy", "xyxyxyxy"]
        self.in_box_format = in_box_format
        self.out_box_format = out_box_format
    
    def __call__(self, data):
        src_h, src_w, ratio_h, ratio_w, dst_h, dst_w = data["shape"]
        bboxes = data["bboxes"]
        if self.in_box_format != self.out_box_format:
            if self.out_box_format == "xywh":
                if self.in_box_format == "xyxyxyxy":
                    bboxes = self.xyxyxyxy2xywh(bboxes)
                elif self.in_box_format == "xyxy":
                    bboxes = self.xyxy2xywh(bboxes)
        bboxes[:, 0::2] *= ratio_w
        bboxes[:, 1::2] *= ratio_h
        bboxes[:, 0::2] /= dst_w
        bboxes[:, 1::2] /= dst_h
        data["bboxes"] = bboxes
        return data
    
    def xyxyxyxy2xywh(self, bboxes):
        new_bboxes = np.zeros([len(bboxes), 4])
        new_bboxes[:, 0] = bboxes[:, 0::2].min()  # x1
        new_bboxes[:, 1] = bboxes[:, 1::2].min()  # y1
        new_bboxes[:, 2] = bboxes[:, 0::2].max() - new_bboxes[:, 0]  # w
        new_bboxes[:, 3] = bboxes[:, 1::2].max() - new_bboxes[:, 1]  # h
        return new_bboxes
    
    def xyxy2xywh(self, bboxes):
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2  # x center
        new_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2  # y center
        new_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # width
        new_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # height
        return new_bboxes
