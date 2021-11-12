import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

from .build import EVALUATOR_REGISTRY
import torch.distributed as dist
import torch.nn.functional as F

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

def vocAP(rec, prec):
    # 11-point mAP, used in voc 2007
    ap = []
    for t in range(0, 11, 1):
        t = t / 10.
        prec_over_rec_thre = prec[rec>=t]
        if len(prec_over_rec_thre) == 0:
            p = 0
        else:
            p = max(prec_over_rec_thre)
        ap.append(p)
    ap = sum(ap) / len(ap)

    return ap

def voc2007mAP(y_pred, y_true):
    nTest = y_true.shape[0]
    nLabel = y_true.shape[1]
    ap = np.zeros(nLabel)
    for i in range(0, nLabel):
        cls_out = y_pred[:,i]
        cls_gt = y_true[:,i]
        R = np.sum(cls_gt==1)
        sort_idx = np.argsort(-cls_out)
        tp = cls_gt[sort_idx] > 0
        fp = cls_gt[sort_idx] < 0
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / R
        prec = np.divide(tp, fp+tp)
        ap[i] = vocAP(rec, prec)
    mAP = np.nanmean(ap)
    return mAP

@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        self._ismulticlass = False
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        self.device = gt.device
        if gt.ndim == 1:
            self._ismulticlass = False
            pred = mo.max(1)[1]
            matches = pred.eq(gt).float()
            self._correct += int(matches.sum().item())
            self._total += gt.shape[0]

            self._y_true.extend(gt.data.cpu().numpy().tolist())
            self._y_pred.extend(pred.data.cpu().numpy().tolist())
        else:
            self._ismulticlass = True
            pred = F.softmax(mo, dim=-1)
            self._y_true.append(gt.data.cpu().numpy())
            self._y_pred.append(pred.data.cpu().numpy())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        if not self._ismulticlass:
            acc = 100.0 * self._correct / self._total
            err = 100.0 - acc
            
            if is_dist_avail_and_initialized() and dist.get_world_size() > 1:
                acc = reduce_tensor(torch.tensor(acc).to(self.device)).cpu().item()
                err = 100.0 - acc
            macro_f1 = 100.0 * f1_score(
                self._y_true,
                self._y_pred,
                average="macro",
                labels=np.unique(self._y_true)
            )

            # The first value will be returned by trainer.test()
            results["accuracy"] = acc
            results["error_rate"] = err
            results["macro_f1"] = macro_f1

            print(
                "=> result\n"
                f"* total: {self._total:,}\n"
                f"* correct: {self._correct:,}\n"
                f"* accuracy: {acc:.2f}%\n"
                f"* error: {err:.2f}%\n"
                f"* macro_f1: {macro_f1:.2f}%"
            )
        else:
            full_logits = np.concatenate(self._y_pred, axis=0)
            full_target = np.concatenate(self._y_true, axis=0)
            acc = voc2007mAP(full_logits, full_target) * 100
            results["accuracy"] = acc
            print(
                "=> result\n"
                f"* accuracy: {acc:.2f}"
            )
        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                # print(
                #     "* class: {} ({})\t"
                #     "total: {:,}\t"
                #     "correct: {:,}\t"
                #     "acc: {:.2f}%".format(
                #         label, classname, total, correct, acc
                #     )
                # )
            mean_acc = np.mean(accs)
            print("* average: {:.2f}%".format(mean_acc))

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print('Confusion matrix is saved to "{}"'.format(save_path))

        self._y_pred = []
        self._y_true = []
        
        return results
