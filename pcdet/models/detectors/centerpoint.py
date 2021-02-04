from .detector3d_template import Detector3DTemplate
from ..model_utils import model_nms_utils
import time
import torch


class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # if not self.training:
        #     self.build_vfe_onnx(batch_dict)
        if not self.training:
            net_time = {}

        for cur_module in self.module_list:
            if not self.training:
                start_time = time.time()
            batch_dict = cur_module(batch_dict)
            if not self.training:
                end_time = time.time()
                net_time[cur_module.model_cfg['NAME']] = (end_time - start_time)*1000   # ms

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            start_time = time.time()
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            end_time = time.time()
            net_time['POST_PROCESSING'] = (end_time - start_time) * 1000  # ms
            recall_dicts['RUN_TIME'] = net_time
            return pred_dicts, recall_dicts

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        pred_dicts = []
        recall_dict = {}
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return batch_dict['pred_dicts'], batch_dict['recall_dict']
        else:
            batch_size = batch_dict['batch_box_preds'].shape[0]
            for index in range(batch_size):
                box_preds = batch_dict['batch_box_preds'][index]
                cls_preds = batch_dict['batch_cls_preds'][index]  # this is the predicted iou
                label_preds = batch_dict['roi_labels'][index]

                scores = torch.sqrt(torch.sigmoid(cls_preds).reshape(-1) * batch_dict['roi_scores'][index].reshape(-1))
                mask = (label_preds != 0).reshape(-1)

                box_preds = box_preds[mask, :]
                scores = scores[mask]
                labels = label_preds[mask]
                # 当前不需要NMS？！
                # currently don't need nms
                if post_process_cfg.get('NMS_CONFIG', None) is not None:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=scores, box_preds=box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    scores = scores[selected]
                    labels = labels[selected]
                    box_preds = box_preds[selected]

                pred_dict = {
                    'pred_boxes': box_preds,
                    'pred_scores': scores,
                    'pred_labels': labels,
                    'pred_keypoints': batch_dict['pred_dicts'][index]['pred_keypoints']
                }
                pred_dicts.append(pred_dict)
            return pred_dicts, recall_dict

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        if self.model_cfg.get('ROI_HEAD', None) is not None:
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss = loss_rpn + loss_rcnn
        else:
            loss = loss_rpn
        return loss, tb_dict, disp_dict
