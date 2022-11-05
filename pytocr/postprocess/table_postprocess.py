import torch
import numpy as np

from .rec_postprocess import AttnLabelDecode


class TableLabelDecode(AttnLabelDecode):
    def __init__(
        self, 
        character_dict_path, 
        merge_no_span_structure=False, 
        **kwargs
    ):
        dict_character = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode("UTF-8").strip("\n").strip("\r\r")
                dict_character.append(line)
        
        if merge_no_span_structure:
            if "<td></td>" not in dict_character:
                dict_character.append("<td></td>")
            if "<td>" in dict_character:
                dict_character.remove("<td>")

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character
        self.td_token = ["<td>", "<td", "<td></td>"]
    
    def __call__(self, preds, batch=None):
        structure_probs = preds["structure_probs"]
        bbox_preds = preds["loc_preds"]
        if isinstance(structure_probs, torch.Tensor):
            structure_probs = structure_probs.detach().cpu().numpy()
        if isinstance(bbox_preds, torch.Tensor):
            bbox_preds = bbox_preds.detach().cpu().numpy()
        shape_list = batch[-1]
        result = self.decode(structure_probs, bbox_preds, shape_list)
        if len(batch) == 1:  # only contains shape
            return result

        label_decode_result = self.decode_label(batch)
        return result, label_decode_result

    def decode(self, structure_probs, bbox_preds, shape_list):
        ignored_tokens = self.get_ignored_tokens()
        end_idx = self.dict[self.end_str]

        structure_idx = structure_probs.argmax(axis=2)
        structure_probs = structure_probs.max(axis=2)

        structure_batch_list = []
        bbox_batch_list = []
        batch_size = len(structure_idx)
        for batch_idx in range(batch_size):
            structure_list = []
            bbox_list = []
            score_list = []
            for idx in range(len(structure_idx[batch_idx])):
                char_idx = int(structure_idx[batch_idx][idx])
                if idx > 0 and char_idx == end_idx:
                    break
                if char_idx in ignored_tokens:
                    continue
                text = self.character[char_idx]
                if text in self.td_token:
                    bbox = bbox_preds[batch_idx, idx]
                    bbox = self._bbox_decode(bbox, shape_list[batch_idx])
                    bbox_list.append(bbox)
                structure_list.append(text)
                score_list.append(structure_probs[batch_idx, idx])
            structure_batch_list.append([structure_list, np.mean(score_list)])
            bbox_batch_list.append(np.array(bbox_list))
        result = {
            "bbox_batch_list": bbox_batch_list, 
            "structure_batch_list": structure_batch_list
        }
        return result
    
    def _bbox_decode(self, bbox, shape):
        # 还原到原图尺寸的坐标
        src_h, src_w, ratio_h, ratio_w, dst_h, dst_w = shape
        bbox[0::2] *= dst_w
        bbox[1::2] *= dst_h
        bbox[0::2] /= ratio_w
        bbox[1::2] /= ratio_h
        return bbox

    def decode_label(self, batch):
        structure_idx = batch[1]
        gt_bbox_list = batch[2]
        shape_list = batch[-1]
        ignored_tokens = self.get_ignored_tokens()
        end_idx = self.dict[self.end_str]

        structure_batch_list = []
        bbox_batch_list = []
        batch_size = len(structure_idx)
        for batch_idx in range(batch_size):
            structure_list = []
            bbox_list = []
            for idx in range(len(structure_idx[batch_idx])):
                char_idx = int(structure_idx[batch_idx][idx])
                if idx > 0 and char_idx == end_idx:
                    break
                if char_idx in ignored_tokens:
                    continue
                structure_list.append(self.character[char_idx])

                bbox = gt_bbox_list[batch_idx][idx]
                if bbox.sum() != 0:
                    bbox = self._bbox_decode(bbox, shape_list[batch_idx])
                    bbox_list.append(bbox)
            structure_batch_list.append(structure_list)
            bbox_batch_list.append(bbox_list)
        result = {
            "bbox_batch_list": bbox_batch_list, 
            "structure_batch_list": structure_batch_list
        }
        return result
        