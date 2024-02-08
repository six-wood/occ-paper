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


def fast_hist(preds, labels, num_classes):
    """Compute the confusion matrix for every batch.

    Args:
        preds (np.ndarray):  Prediction labels of points with shape of
        (num_points, ).
        labels (np.ndarray): Ground truth labels of points with shape of
        (num_points, ).
        num_classes (int): number of classes

    Returns:
        np.ndarray: Calculated confusion matrix.
    """

    k = (labels >= 0) & (labels < num_classes)
    bin_count = np.bincount(num_classes * labels[k].astype(int) + preds[k], minlength=num_classes**2)
    return bin_count[: num_classes**2].reshape(num_classes, num_classes)


def per_class_iou(hist):
    """Compute the per class iou.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        np.ndarray: Calculated per class iou
    """

    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def get_acc(hist):
    """Compute the overall accuracy.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        float: Calculated overall acc
    """

    return np.diag(hist).sum() / hist.sum()


def get_acc_cls(hist):
    """Compute the class average accuracy.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        float: Calculated class average acc
    """

    return np.nanmean(np.diag(hist) / hist.sum(axis=1))


@METRICS.register_module()
class SSCMetric(BaseMetric):
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
        **kwargs,
    ):
        super().__init__(prefix=prefix, collect_device=collect_device)
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix

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
        p, r, iou = 0.0, 0.0, 0.0
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
            tp, fp, fn = self.get_score_completion(data_sample["y_pred"], data_sample["y_true"])
            self.results.append((data_sample["y_pred"], data_sample["y_true"], (tp, fp, fn)))

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

    def seg_eval(self, gt_labels, seg_preds, label2cat, ignore_index, completion_iou, logger=None):
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
        assert len(seg_preds) == len(gt_labels)
        num_classes = len(label2cat)

        hist_list = []
        for i in range(len(gt_labels)):
            gt_seg = gt_labels[i].astype(np.int64)
            pred_seg = seg_preds[i].astype(np.int64)

            # filter out ignored points
            pred_seg[gt_seg == ignore_index] = -1
            gt_seg[gt_seg == ignore_index] = -1

            # calculate one instance result
            hist_list.append(fast_hist(pred_seg, gt_seg, num_classes))

        iou = per_class_iou(sum(hist_list))
        # if ignore_index is in iou, replace it with nan
        if ignore_index < len(iou):
            iou[ignore_index] = np.nan
        miou = np.nanmean(iou)
        acc = get_acc(sum(hist_list))
        acc_cls = get_acc_cls(sum(hist_list))

        header = ["classes"]
        for i in range(len(label2cat)):
            header.append(label2cat[i])
        header.extend(["completion_iou", "miou", "acc", "acc_cls"])

        ret_dict = dict()
        table_columns = [["results"]]
        for i in range(len(label2cat)):
            ret_dict[label2cat[i]] = float(iou[i])
            table_columns.append([f"{iou[i]:.4f}"])
        ret_dict["completion_iou"] = float(completion_iou)
        ret_dict["miou"] = float(miou)
        ret_dict["acc"] = float(acc)
        ret_dict["acc_cls"] = float(acc_cls)

        table_columns.append([f"{completion_iou:.4f}"])
        table_columns.append([f"{miou:.4f}"])
        table_columns.append([f"{acc:.4f}"])
        table_columns.append([f"{acc_cls:.4f}"])

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log("\n" + table.table, logger=logger)

        return ret_dict

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
        ignore_index = self.dataset_meta["ignore_index"]

        gt_ssc_masks = []
        pred_ssc_masks = []
        completion_tp = 0
        completion_fp = 0
        completion_fn = 0
        for eval_ann, sinlge_pred_results, sc_metric in results:
            gt_ssc_masks.append(eval_ann.reshape(-1))
            pred_ssc_masks.append(sinlge_pred_results.reshape(-1))
            tp, fp, fn = sc_metric
            completion_tp += tp
            completion_fp += fp
            completion_fn += fn

        completion_iou = completion_tp / (completion_tp + completion_fp + completion_fn)

        ret_dict = self.seg_eval(
            gt_ssc_masks,
            pred_ssc_masks,
            label2cat,
            ignore_index,
            completion_iou=completion_iou,
            logger=logger,
        )
        return ret_dict
