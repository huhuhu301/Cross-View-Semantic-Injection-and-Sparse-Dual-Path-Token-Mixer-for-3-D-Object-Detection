# Modified by RV-SDTM contributors for this public release; see NOTICE.
from .detector3d_template import Detector3DTemplate


class TransFusion(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        # Cache the AFD loss weight exposed by the dataset-specific backbone.
        self._afd_loss_weight = 0.0
        try:
            if hasattr(self, 'backbone_3d') and hasattr(self.backbone_3d, 'afd_loss_weight'):
                self._afd_loss_weight = float(self.backbone_3d.afd_loss_weight)
            elif 'BACKBONE_3D' in self.model_cfg and hasattr(self.model_cfg.BACKBONE_3D, 'AFD_LOSS_WEIGHT'):
                self._afd_loss_weight = float(self.model_cfg.BACKBONE_3D.AFD_LOSS_WEIGHT)
        except Exception:
            self._afd_loss_weight = 0.0

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}

        loss_trans = batch_dict['loss']
        tb_head = batch_dict.get('tb_dict', {})

        tb_dict = {
            'loss_trans': float(loss_trans.detach().item()) if hasattr(loss_trans, 'item') else float(loss_trans),
            **tb_head
        }

        loss_total = loss_trans

        afd_enabled = (
            hasattr(self, 'backbone_3d')
            and hasattr(self.backbone_3d, 'adaptive_feature_diffusion')
            and bool(self.backbone_3d.adaptive_feature_diffusion)
        )
        afd_w = float(self._afd_loss_weight) if hasattr(self, '_afd_loss_weight') else 0.0

        if afd_enabled and afd_w > 0.0 and hasattr(self.backbone_3d, 'get_loss'):
            loss_afd, tb_afd = self.backbone_3d.get_loss()
            loss_total = loss_total + afd_w * loss_afd

            tb_dict.update(tb_afd)
            try:
                tb_dict['loss_afd'] = float(loss_afd.detach().item())
            except Exception:
                tb_dict['loss_afd'] = float(loss_afd.item())
            tb_dict['loss_total'] = float(loss_total.detach().item())
        else:
            tb_dict['loss_total'] = float(loss_total.detach().item()) if hasattr(loss_total, 'item') else float(loss_total)

        return loss_total, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']
            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )
        return final_pred_dict, recall_dict
