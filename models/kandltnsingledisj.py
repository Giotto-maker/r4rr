import ltn
import torch
import torch.nn as nn

from utils.kand_ltn_loss import KAND_SAT_AGG
from utils.conf import get_device
from utils.args import *

from torch.nn.functional import one_hot
from models.utils.ops import outer_product
from models.utils.utils_problog import *

from shortcut_mitigation.kandinsky.protonet_kand_modules.yolo_modules.yolo import yolo_detect_and_crop_primitives_batch


def get_parser() -> ArgumentParser:
    """Returns the argument parser for the current architecture

    Returns:
        argpars: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class KandLTNSingleDisj(torch.nn.Module):
    """LTN architecture for single Kandinsky primitives using two disentangled backbones"""

    NAME = "kandltnsingledisj"

    def __init__(self, encoder, n_images=3, c_split=(), args=None, nr_facts=6, nr_predicates=9, nr_classes=2):
        super(KandLTNSingleDisj, self).__init__()
        """Initialize method

        Args:
            self: instance
            encoder (nn.Module): encoder
            n_images (int, default=2): number of images
            c_split: concept splits
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        # ^ register the encoders as a parameter
        self.encoder = nn.ModuleList(encoder)

        # number of images, and how to split them
        self.n_images = n_images
        self.n_facts = nr_facts
        self.n_predicates = nr_predicates
        self.nr_classes = nr_classes

        # opt and device
        self.opt = None
        self.device = get_device()
        
        self.w_q, self.and_rule = build_worlds_queries_matrix_KAND(
            self.n_images, self.n_facts, 3, task=args.task
        )
        self.w_q = self.w_q.to(self.device)
        self.and_rule = self.and_rule.to(self.device)

        

    def forward(self, x, concept_extractor, transform, args):
        """
        Forward pass for the model, processing a batch of images to extract and classify shape and color concepts,
        and compute pattern-based predictions.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3 * n_images, 64, 64), containing concatenated images.
            concept_extractor (callable): Function or model used to extract concepts from cropped image regions.
            transform (callable): Transformation function to apply to cropped image regions before concept extraction.
            args (Namespace or dict): Additional arguments required by the concept extractor or cropping function.
        Returns:
            dict: A dictionary containing:
                - "CS" (torch.Tensor): Stacked logits for shape and color concepts for each image in the batch.
                - "pCS" (torch.Tensor): Concatenated probabilities for shape and color concepts after softmax.
                - "YS" (torch.Tensor): One-hot encoded predictions indicating if any pattern (all same, all different, or pair)
                  holds for either shapes or colors across the images in the batch.
        """
        cs, pCs, preds = [], [], []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)  
        for i in range(self.n_images):
            assert xs[i].shape == (x.size(0), 3, 64, 64),\
                f"Expected input shape {(x.size(0), 3, 64, 64)}, but got {xs[i].shape}"
            
            cropped_objects = yolo_detect_and_crop_primitives_batch(xs[i], concept_extractor, transform, args)
            lc_s = self.encoder[0](cropped_objects)
            lc_c = self.encoder[1](cropped_objects)

            # ^ group the logits as LTN expects: turn each (96,3) logits tensor into a (32,9) logits tensor
            lc_s = lc_s.view(xs[i].size(0), self.n_predicates)
            lc_c = lc_c.view(xs[i].size(0), self.n_predicates)
            assert lc_s.shape == (xs[i].size(0), self.n_predicates), f"Expected lc_s shape {(xs[i].size(0), 9)}, but got {lc_s.shape}"
            assert lc_c.shape == (xs[i].size(0), self.n_predicates), f"Expected lc_c shape {(xs[i].size(0), 9)}, but got {lc_c.shape}"

            # ^ concatenate the shapes and colors logits
            lc = torch.cat((lc_s, lc_c), dim=-1)
            assert lc.shape == (xs[i].size(0), 18), f"Expected lc shape {(xs[i].size(0), 18)}, but got {lc.shape}"

            pc = self.normalize_concepts(lc)     
            pred, worlds_prob = self.compute_combined_predicates_probability(pc)
            cs.append(lc), pCs.append(pc), preds.append(pred)

        clen = len(cs[0].shape)
        
        cs = torch.stack(cs, dim=1) if clen > 1 else torch.cat(cs, dim=1)
        pCs = torch.stack(pCs, dim=1) if clen > 1 else torch.cat(pCs, dim=1)
    
        py = self.soft_inference(preds)
        preds = torch.stack(preds, dim=1) if clen > 1 else torch.cat(preds, dim=1)
      
        return {"CS": cs, "pCS": pCs, "YS": py}
    

    # & Symbolic reasoner to get the predictions from the atomic concepts predictions
    '''
        # Query Index Mapping:
        # 
        # | Query Index | Predicate           | Shape Rule     | Color Rule     |
        # |-------------|---------------------|----------------|----------------|
        # | 0           | diffsha & diffcol   | All different  | All different  |
        # | 1           | diffsha & twocol    | All different  | Two-of-a-kind  |
        # | 2           | diffsha & samecol   | All different  | All same       |
        # | 3           | twosha  & diffcol   | Two-of-a-kind  | All different  |
        # | 4           | twosha  & twocol    | Two-of-a-kind  | Two-of-a-kind  |
        # | 5           | twosha  & samecol   | Two-of-a-kind  | All same       |
        # | 6           | samesha & diffcol   | All same       | All different  |
        # | 7           | samesha & twocol    | All same       | Two-of-a-kind  |
        # | 8           | samesha & samecol   | All same       | All same       |
    '''
    def compute_combined_predicates_probability(self, pCs, query=None):
        
        worlds_tensor = outer_product(
            *torch.split(
                pCs.squeeze(1), 3, dim=-1
            )  
        )  

        worlds_prob = worlds_tensor.reshape(-1, 3**self.n_facts)
        query_prob = torch.zeros(size=(len(pCs), self.n_predicates), device=pCs.device)

        for i in range(self.n_predicates):
            query = i
            query_prob[:, i] = self.compute_query(query, worlds_prob).view(-1)

        return query_prob, worlds_prob
    
    
    # & Infer the soft probabilities for the final label
    def soft_inference(self, preds):
        y_worlds = outer_product(*preds).reshape(-1, 9**self.n_images)

        py = torch.zeros(size=(len(preds[0]), self.nr_classes), device=preds[0].device)

        for i in range(self.nr_classes):
            and_rule = self.and_rule[:, i]
            query_prob = torch.sum(and_rule * y_worlds, dim=1, keepdim=True)

            py[:, i] = query_prob.view(-1)

        return py


    # & Computes query probability given the worlds probability P(w).
    def compute_query(self, query, worlds_prob):
        w_q = self.w_q[:, query]
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
        return query_prob


    # & Loss function for the LTN architecture
    def get_loss(self, args):
        _and, _implies = None, None
        if args.and_op == "Godel":
            _and = ltn.fuzzy_ops.AndMin()
        elif args.and_op == "Prod":
            _and = ltn.fuzzy_ops.AndProd()
        else:
            _and = ltn.fuzzy_ops.AndLuk()
        if args.or_op == "Godel":
            Or = ltn.Connective(ltn.fuzzy_ops.OrMax())
        elif args.or_op == "Prod":
            Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        else:
            Or = ltn.Connective(ltn.fuzzy_ops.OrLuk())
        if args.imp_op == "Godel":
            _implies = ltn.fuzzy_ops.ImpliesGodel()
        elif args.imp_op == "Prod":
            _implies = ltn.fuzzy_ops.ImpliesReichenbach()
        elif args.imp_op == "Luk":
            _implies = ltn.fuzzy_ops.ImpliesLuk()
        elif args.imp_op == "Goguen":
            _implies = ltn.fuzzy_ops.ImpliesGoguen()
        else:
            _implies = ltn.fuzzy_ops.ImpliesKleeneDienes()

        And = ltn.Connective(_and)
        Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        Forall = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=args.p), quantifier="f"
        )
        Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(and_op=_and, implies_op=_implies))
        return KAND_SAT_AGG(And, Or, Not, Forall, Equiv)
    

    # & Normalize concepts method to get pCs for cs
    def normalize_concepts(self, z):
        def soft_clamp(h, dim=-1):
            h = torch.nn.Softmax(dim=dim)(h)
            eps = 1e-5
            h = h + eps
            with torch.no_grad():
                Z = torch.sum(h, dim=dim, keepdim=True)
            h = h / Z
            return h
        pCi = torch.split(z, 3, dim=-1)

        norm_concepts = torch.cat(
            [soft_clamp(c) for c in pCi], dim=-1
        ) 

        return norm_concepts
    

    # override of to
    def to(self, device):
        super().to(device)
        self.w_q = self.w_q.to(device)
        self.and_rule = self.and_rule.to(device)