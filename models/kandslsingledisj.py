import torch
import torch.nn as nn

from utils.args import *
from utils.conf import get_device
from utils.losses import *

from models.cext import CExt
from models.utils.ops import outer_product
from models.utils.utils_problog import build_worlds_queries_matrix_KAND

from utils.semantic_loss import KANDINSKY_SL

from shortcut_mitigation.kandinsky.protonet_kand_modules.yolo_modules.yolo import yolo_detect_and_crop_primitives_batch


def get_parser() -> ArgumentParser:
    """Returns the argument parser for this architecture"""
    parser = ArgumentParser(description="Learning via Kandinsky Concept Extractor.")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class KandSLSingleDisj(CExt):
    """SL architecture for single Kandinsky primitives using two disentangled backbones"""

    NAME = "kandslsingledisj"

    # & Ok 
    def __init__(self, encoder, n_images=3, c_split=(), args=None, n_facts=6, nr_classes=2):
        """
        Initialize the Kandinsky architecture with semantic loss.
        
        Args:
            encoder (nn.Module): Encoder network
            n_images (int): Number of subimages per image (default: 3)
            c_split: Concept split configuration
            args: Command-line arguments
            n_facts (int): Number of concepts (default: 6 for 3 shapes × 2 colors)
            nr_classes (int): Number of output classes (default: 2 for binary classification)
        """
        super(KandSLSingleDisj, self).__init__(encoder=encoder, n_images=n_images, c_split=c_split)

        # ^ register the encoders as a parameter
        self.encoder = nn.ModuleList(encoder)

        # Logical world-query matrix
        self.n_facts = n_facts
        self.logic, self.and_rule = build_worlds_queries_matrix_KAND(
            n_images=self.n_images, 
            n_concepts=self.n_facts, 
            n_poss=3,
            task="patterns")
        self.nr_classes = nr_classes
        self.n_predicates = 9

        assert self.n_images == 3, "KandinskySL expects 3 subimages"
        assert self.n_facts == 6, "KandinskySL expects 3 shapes × 2 colors"
        assert self.nr_classes == 2, "KandinskySL only supports binary classification"
        assert self.logic.shape == (729, 9), f"Expected shape (729, 9), but got {self.logic.shape}"

        # Optimizer and device configuration
        self.opt = None
        self.device = get_device()
        self.logic = self.logic.to(self.device)
        self.and_rule = self.and_rule.to(self.device)

    # & Forward method
    def forward(self, x, concept_extractor, transform, args):
        """
        Performs a forward pass through the model, extracting and processing concepts from input images.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels * n_images, height, width).
            concept_extractor (callable): Function or model used to extract concepts from images.
            transform (callable): Transformation function to apply to cropped objects.
            args (Namespace or dict): Additional arguments required for concept extraction and cropping.
        Returns:
            dict: A dictionary containing:
                - "CS" (torch.Tensor): Stacked logits for shapes and colors, shape (batch_size, n_images, 18).
                - "YS" (torch.Tensor): Predicted class probabilities, shape (batch_size, nr_classes).
                - "pCS" (torch.Tensor): Normalized concept probabilities, shape (batch_size, n_images, ...).
                - "gpreds" (Any): Aggregated predicate and label probabilities as computed by `compute_predicates_label_probability`.
        Raises:
            AssertionError: If intermediate tensor shapes do not match expected dimensions.
        """
        cs, pCs, preds = [], [], []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            assert xs[i].shape == (x.size(0), 3, 64, 64), f"Expected input shape {(x.size(0), 3, 64, 64)}, but got {xs[i].shape}"
            cropped_objects = yolo_detect_and_crop_primitives_batch(xs[i], concept_extractor, transform, args)
            lc_s = self.encoder[0](cropped_objects)
            lc_c = self.encoder[1](cropped_objects)

            # ^ group the logits as SL expects: turn each (96,3) logits tensor into a (32,9) logits tensor
            lc_s = lc_s.view(xs[i].size(0), self.n_predicates)
            lc_c = lc_c.view(xs[i].size(0), self.n_predicates)
            assert lc_s.shape == (xs[i].size(0), self.n_predicates), f"Expected lc_s shape {(xs[i].size(0), 9)}, but got {lc_s.shape}"
            assert lc_c.shape == (xs[i].size(0), self.n_predicates), f"Expected lc_c shape {(xs[i].size(0), 9)}, but got {lc_c.shape}"

            # ^ concatenate the shapes and colors logits
            lc = torch.cat((lc_s, lc_c), dim=-1)
            assert lc.shape == (xs[i].size(0), 18), f"Expected lc shape {(xs[i].size(0), 18)}, but got {lc.shape}"
            
            pc = self.normalize_concepts(lc)
            pred, _ = self.compute_combined_predicates_probability(pc)
            cs.append(lc), pCs.append(pc), preds.append(pred)
        
        clen = len(cs[0].shape)
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1) # torch.Size([32, 3, 18])
        pCs = torch.stack(pCs, dim=1) if clen == 2 else torch.cat(pCs, dim=1) # torch.Size([32, 3, 3, 2, 3])
        
        # Aggregate concepts across batch size to get predictions
        batch_size = x.size(0)
        py = self.soft_inference(preds)

        # For consistency, one may include assertions on the shapes:
        assert cs.shape[0] == batch_size and cs.shape[1] == self.n_images, "Unexpected cs shape"
        assert py.shape == (batch_size, self.nr_classes), f"Expected YS shape {(batch_size, self.nr_classes)}, got {py.shape}"

        preds = torch.stack(preds, dim=1) if clen > 1 else torch.cat(preds, dim=1)
        aggregated_preds_and_label = self.compute_predicates_label_probability(preds, py, x.size(0))

        # Return predictions
        return {"CS": cs, "YS": py, "pCS": pCs, "gpreds": aggregated_preds_and_label}
    

    # & Normalize concepts method to get pCs for cs
    def normalize_concepts(self, z):
        def soft_clamp(h, dim=-1):
            h = nn.Softmax(dim=dim)(h)
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
    

    # & Combine queries method
    def soft_inference(self, preds):
        y_worlds = outer_product(*preds).reshape(-1, 9**self.n_images)

        py = torch.zeros(size=(len(preds[0]), self.nr_classes), device=preds[0].device)

        for i in range(self.nr_classes):
            and_rule = self.and_rule[:, i]
            query_prob = torch.sum(and_rule * y_worlds, dim=1, keepdim=True)
            py[:, i] = query_prob.view(-1)

        return py


    # & Computes query probability given the worlds probability P(w).
    # Select the column of logic matrix corresponding to the current query and 
    # compute query probability by summing the probability of all the worlds where the query is true
    def compute_query(self, query, worlds_prob):
        logic = self.logic[:, query]
        query_prob = torch.sum(logic * worlds_prob, dim=1, keepdim=True)

        return query_prob
    

    # & Given the combined probs from compute_combined_predicates_probability, get probs for each predicate
    def compute_predicates_label_probability(self, high_concepts, label, batch_size):
        aggregated_preds = []  
        for pred in high_concepts:  # each pred is (batch, 9)
            p_diff_shape = pred[:, 0] + pred[:, 1] + pred[:, 2]
            p_twoshape   = pred[:, 3] + pred[:, 4] + pred[:, 5]
            p_same_shape = pred[:, 6] + pred[:, 7] + pred[:, 8]
            
            p_diff_color = pred[:, 0] + pred[:, 3] + pred[:, 6]
            p_twocolor   = pred[:, 1] + pred[:, 4] + pred[:, 7]
            p_same_color = pred[:, 2] + pred[:, 5] + pred[:, 8]
            
            image_pred = torch.stack([p_diff_shape, p_twoshape, p_same_shape,
                                      p_diff_color, p_twocolor, p_same_color], dim=1)  # (batch, 6)
            aggregated_preds.append(image_pred)
        
        # Now, 'aggregated_preds' is a list of 3 tensors, each of shape (batch,6)
        images_tensor = torch.stack(aggregated_preds, dim=1).permute(1, 0, 2)
        assert images_tensor.shape == (batch_size, 3, 6), \
            f"Expected images_tensor shape {batch_size, 3, 6}, but got {images_tensor.shape}"
        
        # Pad the final label with 0s to match the shape of the constrained variables
        label_tensor = torch.cat([label, torch.zeros(label.shape[0], 4, device=label.device)], dim=1)  # (batch,6)
        
        assert images_tensor.shape == (batch_size, 3, 6), \
            f"Expected images_tensor shape {batch_size, 3, 6}, but got {images_tensor.shape}"
        assert label_tensor.unsqueeze(1).shape == (batch_size, 1, 6), \
            f"Expected label_tensor shape {batch_size, 1, 6}, but got {label_tensor.unsqueeze(1).shape}"
        
        final_tensor = torch.cat([images_tensor, label_tensor.unsqueeze(1)], dim=1)  # (batch, 4, 6)
        return final_tensor

    
    # & Semantic loss method
    def get_loss(self, args):
        if args.dataset in ["kandinsky"]:
            return KANDINSKY_SL(KAND_Cumulative, args)
        else:
            raise NotImplementedError("Dataset not supported for KandinskySL")


    # override of to
    def to(self, device):
        super().to(device)
        self.logic = self.logic.to(device)
        self.and_rule = self.and_rule.to(device)
