"""
Part of the code is taken from https://github.com/waterljwant/SSC/blob/master/sscMetrics.py
"""
import numpy as np
import os.path as osp
import tempfile
from typing import Dict, Optional, Sequence

import mmcv
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.registry import METRICS
from mmengine.logging import print_log
from terminaltables import AsciiTable


def get_iou(iou_sum, cnt_class):
    _C = iou_sum.shape[0]  # 12
    iou = np.zeros(_C, dtype=np.float32)  # iou for each class
    for idx in range(_C):
        iou[idx] = iou_sum[idx] / cnt_class[idx] if cnt_class[idx] else 0

    mean_iou = np.sum(iou[1:]) / np.count_nonzero(cnt_class[1:])
    return iou, mean_iou


def get_accuracy(predict, target, weight=None):  # 0.05s
    _bs = predict.shape[0]  # batch size
    _C = predict.shape[1]  # _C = 12
    target = np.int32(target)
    target = target.reshape(_bs, -1)  # (_bs, 60*36*60) 129600
    predict = predict.reshape(_bs, _C, -1)  # (_bs, _C, 60*36*60)
    predict = np.argmax(predict, axis=1)  # one-hot: _bs x _C x 60*36*60 -->  label: _bs x 60*36*60.

    correct = predict == target  # (_bs, 129600)
    if weight:  # 0.04s, add class weights
        weight_k = np.ones(target.shape)
        for i in range(_bs):
            for n in range(target.shape[1]):
                idx = 0 if target[i, n] == 255 else target[i, n]
                weight_k[i, n] = weight[idx]
        correct = correct * weight_k
    acc = correct.sum() / correct.size
    return acc


class SSCompute:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def hist_info(self, n_cl, pred, gt):
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl**2).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    @staticmethod
    def compute_score(hist, correct, labeled):
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_IU = np.nanmean(iu)
        mean_IU_no_back = np.nanmean(iu[1:])
        freq = hist.sum(1) / hist.sum()
        freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
        mean_pixel_acc = correct / labeled if labeled != 0 else 0

        return iu, mean_IU, mean_IU_no_back, mean_pixel_acc

    def add_batch(self, y_pred, y_true, nonempty=None, nonsurface=None):
        self.count += 1
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)

        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(y_pred, y_true, mask)
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum

    def get_stats(self):
        if self.completion_tp != 0:
            precision = self.completion_tp / (self.completion_tp + self.completion_fp)
            recall = self.completion_tp / (self.completion_tp + self.completion_fn)
            iou = self.completion_tp / (self.completion_tp + self.completion_fp + self.completion_fn)
        else:
            precision, recall, iou = 0, 0, 0
        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)
        return {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "iou_ssc": iou_ssc,
            "iou_ssc_mean": np.mean(iou_ssc[1:]),
        }

    def reset(self):
        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.tps = np.zeros(self.n_classes)
        self.fps = np.zeros(self.n_classes)
        self.fns = np.zeros(self.n_classes)

        self.hist_ssc = np.zeros((self.n_classes, self.n_classes))
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.precision = 0
        self.recall = 0
        self.iou = 0
        self.count = 1e-8
        self.iou_ssc = np.zeros(self.n_classes, dtype=np.float32)
        self.cnt_class = np.zeros(self.n_classes, dtype=np.float32)

    def get_score_completion(self, predict, target, nonempty=None):
        predict = np.copy(predict)
        target = np.copy(target)

        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = np.zeros(predict.shape)
        b_true = np.zeros(target.shape)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]

            tp = np.array(np.where(np.logical_and(y_true == 1, y_pred == 1))).size
            fp = np.array(np.where(np.logical_and(y_true != 1, y_pred == 1))).size
            fn = np.array(np.where(np.logical_and(y_true == 1, y_pred != 1))).size
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        return tp_sum, fp_sum, fn_sum

    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        target = np.copy(target)
        predict = np.copy(predict)
        _bs = predict.shape[0]  # batch size
        _C = self.n_classes  # _C = 12
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, 129600), 60*36*60=129600

        tp_sum = np.zeros(_C, dtype=np.int32)  # tp
        fp_sum = np.zeros(_C, dtype=np.int32)  # fp
        fn_sum = np.zeros(_C, dtype=np.int32)  # fn

        for idx in range(_bs):
            y_true = target[idx, :]  # GT
            y_pred = predict[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_pred = y_pred[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]
                y_true = y_true[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]
            for j in range(_C):  # for each class
                tp = np.array(np.where(np.logical_and(y_true == j, y_pred == j))).size
                fp = np.array(np.where(np.logical_and(y_true != j, y_pred == j))).size
                fn = np.array(np.where(np.logical_and(y_true == j, y_pred != j))).size

                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn

        return tp_sum, fp_sum, fn_sum


@METRICS.register_module()
class SscMetric(BaseMetric):
    """3D semantic segmentation evaluation metric.

    Args:
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default: None.
    """

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        pklfile_prefix: str = None,
        submission_prefix: str = None,
        free_index: int = 0,
        ignore_index: int = 255,
        num_classes: int = 20,
        **kwargs,
    ):
        super().__init__(prefix=prefix, collect_device=collect_device)
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        self.free_index = free_index
        self.ignore_index = ignore_index
        self.ssc_compute = SSCompute(num_classes)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            self.ssc_compute.add_batch(data_sample["y_pred"], data_sample["y_true"])

    def log_show(self, stats, label2cat, ignore_index, logger=None):
        """Semantic Segmentation  Evaluation.

        Evaluate the result of the Semantic Segmentation.

        Args:
            gt_labels (list[torch.Tensor]): Ground truth labels.
            seg_preds  (list[torch.Tensor]): Predictions.
            label2cat (dict): Map from label to category name.
            ignore_index (int): Index that will be ignored in evaluation.
            logger (logging.Logger | str, optional): The way to print the mAP
                summary. See `mmdet.utils.print_log()` for details. Default: None.

        Returns:
            dict[str, float]: Dict of results.
        """
        precision = stats["precision"]
        recall = stats["recall"]
        iou = stats["iou"]
        iou_ssc = stats["iou_ssc"]
        miou = stats["iou_ssc_mean"]

        header = ["classes"]
        for i in range(len(label2cat)):
            header.append(label2cat[i])
        header.extend(["iou", "miou", "precision", "recall"])

        ret_dict = dict()
        table_columns = [["results"]]
        for i in range(len(label2cat)):
            ret_dict[label2cat[i]] = float(iou_ssc[i])
            table_columns.append([f"{iou_ssc[i]:.4f}"])
        ret_dict["iou"] = float(iou)
        ret_dict["miou"] = float(miou)
        ret_dict["precision"] = float(precision)
        ret_dict["recall"] = float(recall)

        table_columns.append([f"{iou:.4f}"])
        table_columns.append([f"{miou:.4f}"])
        table_columns.append([f"{precision:.4f}"])
        table_columns.append([f"{recall:.4f}"])

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log("\n" + table.table, logger=logger)

        return ret_dict

    def format_results(self, results):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """

        submission_prefix = self.submission_prefix
        if submission_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            submission_prefix = osp.join(tmp_dir.name, "results")
        mmcv.mkdir_or_exist(submission_prefix)
        ignore_index = self.dataset_meta["ignore_index"]
        # need to map network output to original label idx
        cat2label = np.zeros(len(self.dataset_meta["label2cat"])).astype(np.int64)
        for original_label, output_idx in self.dataset_meta["label2cat"].items():
            if output_idx != ignore_index:
                cat2label[output_idx] = original_label

        for i, (eval_ann, result) in enumerate(results):
            sample_idx = eval_ann["point_cloud"]["lidar_idx"]
            pred_sem_mask = result["semantic_mask"].numpy().astype(np.int64)
            pred_label = cat2label[pred_sem_mask]
            curr_file = f"{submission_prefix}/{sample_idx}.txt"
            np.savetxt(curr_file, pred_label, fmt="%d")

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            return None

        label2cat = self.dataset_meta["label2cat"]
        if "ignore_index" in self.dataset_meta:
            ignore_index = self.dataset_meta["ignore_index"]
        else:
            ignore_index = self.ignore_index

        stats = self.ssc_compute.get_stats()

        ret_dict = self.log_show(
            stats,
            label2cat,
            ignore_index,
            logger=logger,
        )
        return ret_dict
