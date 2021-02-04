from .detector3d_template import Detector3DTemplate
import time


class PointPillar(Detector3DTemplate):
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

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
